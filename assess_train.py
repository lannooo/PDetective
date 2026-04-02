import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import yaml
import logging
import lightning as L
import torch
import shutil

from argparse import Namespace
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from module.config import config
from module.util.utils import import_class, import_class_from_path, get_mask
from module.util.vis import EvalResultCache
from module.data.dataset import DataModule
from module.util.metric import UtteranceMetric, FrameMetric, BoundaryMetric
from module.nn.loss import *
from module.callbacks.sam import SAM

BASE_EXP_DIR = config.get("path.exp")

class ConfigNamespace(Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = ConfigNamespace(**value)
            setattr(self, key, value)
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


class CustomModule(L.LightningModule):
    def __init__(self, args, configs, tblogger):
        super().__init__()
        self.save_hyperparameters(ignore=['tblogger'])
        self.tblogger = tblogger
        self.args = args
        self.configs = configs

        # load different models from different Model path
        run_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.py')
        if os.path.exists(run_path):
            model_cls, package_path = import_class_from_path(configs.model, run_path)
        else:
            model_cls, package_path = import_class(configs.model)
    
        self.model = model_cls(args, configs)
        self.model_path = package_path
        print("Load model from:", package_path)

        # Define masked version of loss function
        # BAM: ce-loss, boundary-balance-loss
        # TDL: ce-loss, embedding_loss
        # AASIST / RawNet2: bce/ce
        self.configure_objectives()

        # Define metric listeners
        self.utt_metric = [UtteranceMetric()]
        self.frame_metric = [FrameMetric()]
        self.bdy_metric = [BoundaryMetric()]
        self.current_dataset_idx = -1


    def setup(self, stage: str):
        # copy training script and model script to log dir
        shutil.copyfile(os.path.abspath(__file__), f"{self.tblogger.log_dir}/run.py")
        shutil.copyfile(os.path.abspath(self.model_path), f"{self.tblogger.log_dir}/model.py")
        # configure the logger
        self.console_logger = logging.getLogger(f"lightning.pytorch.{stage}")
        file_handler = logging.FileHandler(f"{self.tblogger.log_dir}/{stage}.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.console_logger.addHandler(file_handler)
    

    def configure_objectives(self):
        loss_configs = self.configs.objective
        # get all the loss keys, note that the loss_configs is ConfigNamespace
        # if isinstance(loss_configs, ConfigNamespace):
        #     print("Convert loss configs to dict")
        loss_configs = loss_configs.to_dict()
        losses = {}
        for key, cfg in loss_configs.items():
            loss_name, loss, loss_weight = make_loss_fn(cfg, key)
            losses[loss_name] = {
                "fn": loss,
                "weight": loss_weight
            }
        self.objectives = list(losses.keys())
        self.losses = losses
        self.objective_handlers = {
            "frame": self._handle_frame_objective,
            "multireso": self._handle_multireso_objective,
            "boundary": self._handle_boundary_objective,
            "utterance": self._handle_utterance_objective,
            "embedding": self._handle_embedding_objective,
            "tcl": self._handle_tcl_objective,
            "reg": self._handle_reg_objective,
        }

        unsupported = [objective for objective in self.objectives if objective not in self.objective_handlers]
        if unsupported:
            raise ValueError(f"Unsupported objectives: {unsupported}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs.optim.learning_rate,
            weight_decay=self.configs.optim.weight_decay,
        )
        if hasattr(self.configs.scheduler, "type"):
            scheduler_type = self.configs.scheduler.type
            if scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                       T_max=self.configs.scheduler.step_size * self.args.max_epochs, 
                                                                       eta_min=self.configs.scheduler.lr_min)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.configs.scheduler.step_size, 
                                                        gamma=self.configs.scheduler.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


    def _log_objective_loss(self, stage, name, loss, prog_bar, log_detail):
        if log_detail:
            self.log(f"{stage}/{name}", loss.item(), prog_bar=prog_bar, on_epoch=True)


    def _get_loss_config(self, objective_name):
        return self.losses[objective_name]["fn"], self.losses[objective_name]["weight"]


    def _compute_frame_score(self, frame_pred):
        if len(frame_pred.shape) == 3 and frame_pred.shape[2] == 2:
            return frame_pred[:, :, 1] - frame_pred[:, :, 0]
        if len(frame_pred.shape) == 2:
            return frame_pred
        raise ValueError(f"Invalid frame prediction shape: {frame_pred.shape}")


    def _handle_frame_objective(self, outputs, stage, dataloader_idx, update_metric, log_detail, prog_bar):
        loss_fn, weight = self._get_loss_config("frame")
        frm_pred, frm_target, frm_length = outputs["frame_pred"], outputs["frame_target"], outputs["frame_length"]
        frm_score = self._compute_frame_score(frm_pred)

        if update_metric:
            self.frame_metric[dataloader_idx].update(stage, frm_score, frm_target, frm_length)

        if stage == 'test':
            return None

        frame_mask = get_mask(frm_target, frm_length)
        if isinstance(loss_fn, MaskCrossEnrtopyLoss):
            loss = loss_fn(frm_pred.transpose(-1, -2), frm_target, frame_mask)
        elif isinstance(loss_fn, MaskBCELoss):
            loss = loss_fn(frm_pred, frm_target.float(), frame_mask)
        else:
            loss = loss_fn(frm_pred, frm_target, frame_mask)

        self._log_objective_loss(stage, "loss_frm", loss, prog_bar, log_detail)
        return weight * loss


    def _handle_multireso_objective(self, outputs, stage, dataloader_idx, update_metric, log_detail, prog_bar):
        loss_fn, weight = self._get_loss_config("multireso")
        multi_frm_pred, multi_frm_target, multi_frm_length = outputs["frame_pred"], outputs["frame_target"], outputs["frame_length"]
        frm_pred, frm_target, frm_length = multi_frm_pred[-1], multi_frm_target[-1], multi_frm_length[-1]
        frm_score = self._compute_frame_score(frm_pred)

        if update_metric:
            self.frame_metric[dataloader_idx].update(stage, frm_score, frm_target, frm_length)

        if stage == 'test':
            return None

        loss = 0.0
        for pred, target, length in zip(multi_frm_pred, multi_frm_target, multi_frm_length):
            frame_mask = get_mask(target, length)
            loss += loss_fn(pred, target, frame_mask)

        if 'utt_pred' in outputs and outputs['utt_pred'] is not None:
            utt_pred, utt_target = outputs['utt_pred'], outputs['utt_target']
            loss += loss_fn(utt_pred, utt_target)

        self._log_objective_loss(stage, "loss_frm", loss, prog_bar, log_detail)
        return weight * loss


    def _handle_boundary_objective(self, outputs, stage, dataloader_idx, update_metric, log_detail, prog_bar):
        loss_fn, weight = self._get_loss_config("boundary")
        bdr_pred, bdr_target, bdr_length = outputs["boundary_pred"], outputs["boundary_target"], outputs["boundary_length"]

        if update_metric:
            self.bdy_metric[dataloader_idx].update(stage, bdr_pred, bdr_target.long(), bdr_length)

        if stage == 'test':
            return None

        bdy_mask = get_mask(bdr_target, bdr_length)
        loss = loss_fn(bdr_pred, bdr_target, bdy_mask)
        self._log_objective_loss(stage, "loss_bdy", loss, prog_bar, log_detail)
        return weight * loss


    def _handle_utterance_objective(self, outputs, stage, dataloader_idx, update_metric, log_detail, prog_bar):
        loss_fn, weight = self._get_loss_config("utterance")
        utt_pred, utt_target = outputs["utt_pred"], outputs["utt_target"]
        utt_score = utt_pred[:, 1] - utt_pred[:, 0]

        if update_metric:
            self.utt_metric[dataloader_idx].update(stage, utt_score, utt_target)

        if stage == 'test':
            return None

        loss = loss_fn(utt_pred, utt_target)
        self._log_objective_loss(stage, "loss_utt", loss, prog_bar, log_detail)
        return weight * loss


    def _handle_embedding_objective(self, outputs, stage, dataloader_idx, update_metric, log_detail, prog_bar):
        del dataloader_idx, update_metric
        loss_fn, weight = self._get_loss_config("embedding")
        if stage == 'test':
            return None

        frm_length = outputs["frame_length"]
        frm_target = outputs["frame_target"]
        embeddings = outputs["frame_emb"]
        pad_mask = get_mask(frm_target, frm_length)
        loss = loss_fn(embeddings, frm_target, pad_mask)
        self._log_objective_loss(stage, "loss_emb", loss, prog_bar, log_detail)
        return weight * loss


    def _handle_tcl_objective(self, outputs, stage, dataloader_idx, update_metric, log_detail, prog_bar):
        del dataloader_idx, update_metric
        loss_fn, weight = self._get_loss_config("tcl")
        if stage == 'test':
            return None

        embeddings = outputs["frame_emb"]
        frm_target = outputs["emb_frame_target"]
        frm_length = outputs["emb_frame_length"]
        pad_mask = get_mask(frm_target, frm_length)
        loss = loss_fn(embeddings, frm_target, pad_mask)
        self._log_objective_loss(stage, "loss_tcl", loss, prog_bar, log_detail)
        return weight * loss


    def _handle_reg_objective(self, outputs, stage, dataloader_idx, update_metric, log_detail, prog_bar):
        del dataloader_idx, update_metric
        _, weight = self._get_loss_config("reg")
        if stage != 'train':
            return None

        reg_loss = outputs.get("reg_loss")
        self._log_objective_loss(stage, "loss_reg", reg_loss, prog_bar, log_detail)
        return weight * reg_loss


    def _cache_test_outputs(self, outputs):
        utt_ids = outputs['utt_id']

        frame_labels = outputs['frame_target'].detach().cpu()
        if isinstance(frame_labels, list):
            frame_labels = frame_labels[-1]
        frame_pred = outputs['frame_pred'].detach().cpu()
        if isinstance(frame_pred, list):
            frame_pred = frame_pred[-1]
        frame_length = outputs["frame_length"].detach().cpu()
        if isinstance(frame_length, list):
            frame_length = frame_length[-1]

        frame_embs = outputs['frame_emb'].detach().cpu() if self.args.visualize else None
        embedding_frame_labels = outputs["emb_frame_target"].detach().cpu() if self.args.visualize else None
        embedding_frame_length = outputs["emb_frame_length"].detach().cpu() if self.args.visualize else None
        frame_scores = self._compute_frame_score(frame_pred)
        if utt_ids is not None:
            self.eval_cache.write_cache(
                utt_ids,
                frame_scores,
                frame_labels,
                frame_length,
                frame_embs,
                embedding_frame_labels,
                embedding_frame_length,
            )
    

    def compute_objective(self, batch, stage, dataloader_idx=0, update_metric=True, log_detail=True):
        outputs = self.model.forward_x(batch)
        loss_total = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        prog_bar = True if stage != 'test' else False
        for objective_name in self.objectives:
            objective_loss = self.objective_handlers[objective_name](
                outputs,
                stage,
                dataloader_idx,
                update_metric,
                log_detail,
                prog_bar,
            )
            if objective_loss is not None:
                loss_total += objective_loss

        if stage == 'test' and update_metric:
            self._cache_test_outputs(outputs)

        # total loss
        if stage != 'test' and log_detail:
            self.log(f'{stage}_loss', loss_total.item(), on_epoch=True)

        return loss_total


    def compute_metric(self, stage, dataloader_idx=0, temperary=False):
        metric_states = []
        if "frame" in self.objectives or "multireso" in self.objectives:
            eer, thres1, precision, recall, thres2, auc = self.frame_metric[dataloader_idx].report(stage)
            if not temperary:
                self.log(f'{stage}-{dataloader_idx}/frm/eer', eer)
                self.log(f'{stage}-{dataloader_idx}/frm/thres', thres1)
                self.log(f'{stage}-{dataloader_idx}/frm/precision', precision)
                self.log(f'{stage}-{dataloader_idx}/frm/recall', recall)
                self.log(f'{stage}-{dataloader_idx}/frm/auroc', auc)
                self.frame_metric[dataloader_idx].reset(stage=stage)
            metric_states.append(f"Frame - EER: {100*eer:.3f} %, Thres: {thres1:.3f}"
                                 f", Precision: {precision:.4f}, Recall: {recall:.4f}, Thres: {thres2:.3f}"
                                 f", AUROC: {auc:.4f}")
        if "boundary" in self.objectives:
            eer_b, threshold_b = self.bdy_metric[dataloader_idx].report(stage)
            if not temperary:
                self.log(f'{stage}-{dataloader_idx}/bdy/eer', eer_b)
                self.log(f'{stage}-{dataloader_idx}/bdy/thres', threshold_b)
                self.bdy_metric[dataloader_idx].reset(stage=stage)
            metric_states.append(f"Boundary - EER: {100*eer_b:.3f} %, Thres: {threshold_b:.3f}")
        if "utterance" in self.objectives:
            eer, thres1, precision, recall, thres2, auc = self.utt_metric[dataloader_idx].report(stage)
            if not temperary:
                self.log(f'{stage}-{dataloader_idx}/utt/eer', eer)
                self.log(f'{stage}-{dataloader_idx}/utt/thres', thres1)
                self.log(f'{stage}-{dataloader_idx}/utt/precision', precision)
                self.log(f'{stage}-{dataloader_idx}/utt/recall', recall)
                self.log(f'{stage}-{dataloader_idx}/utt/auroc', auc)
                self.utt_metric[dataloader_idx].reset(stage=stage)
            metric_states.append(f"Utterance - EER: {100*eer:.3f} %, Thres: {thres1:.3f}"
                                 f", Precision: {precision:.4f}, Recall: {recall:.4f}, Thres: {thres2:.3f}"
                                 f", AUROC: {auc:.4f}")
        if stage != 'train':
            if stage == 'test':
                dataset_key = self.trainer.datamodule.test_dataset_names[dataloader_idx]
            else:
                dataset_key = f"dev:{dataloader_idx}"
            self.console_logger.info(f'[{stage}] Epoch: {self.current_epoch} Dataset: {dataset_key}')
            self.console_logger.info('-' * 20)
            for state in metric_states:
                self.console_logger.info(state)
            self.console_logger.info('-' * 20)

    
    def training_step(self, batch, batch_idx):
        if hasattr(self.model, 'set_training_progress'):
            total_steps = getattr(self.trainer, 'estimated_stepping_batches', 0)
            self.model.set_training_progress(self.global_step, total_steps)
        return self.compute_objective(batch, stage='train')
    
    def on_train_epoch_end(self):
        self.compute_metric(stage='train')

    def validation_step(self, batch, batch_idx):
        return self.compute_objective(batch, stage='validate')

    def on_validation_epoch_end(self):
        self.compute_metric(stage='validate')
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_key = self.trainer.datamodule.test_dataset_names[dataloader_idx]
        if self.current_dataset_idx == -1:
            self.current_dataset_idx = dataloader_idx      # should be 0
            self.eval_cache.set_cache_file(dataset_key, dataloader_idx)
        elif dataloader_idx != self.current_dataset_idx:   # there are multiple dataloaders
             # compute metric for the previous dataset
            self.compute_metric(stage='test', dataloader_idx=self.current_dataset_idx, temperary=True)

            self.console_logger.info(f'Test dataset changed to {dataloader_idx}, add new metric tracker')
            self.frame_metric.append(FrameMetric())
            self.utt_metric.append(UtteranceMetric())
            self.bdy_metric.append(BoundaryMetric())
            
            self.current_dataset_idx = dataloader_idx
            self.eval_cache.save_cache()  # save cache for the previous dataset
            self.eval_cache.set_cache_file(dataset_key, dataloader_idx)  # set cache file for the new dataset

        self.compute_objective(batch, stage='test', dataloader_idx=dataloader_idx)
        # if batch_idx % 100 == 0:
        #     import objgraph
        #     # pip install objgraph
        #     objgraph.show_most_common_types(limit=10)

    
    def on_test_epoch_start(self):
        self.console_logger.info(f'Test epoch start...')
        self.eval_cache = EvalResultCache(
            os.path.join(self.tblogger.log_dir, 'cache'), self.args.visualize
        )
    
    def on_test_epoch_end(self):
        self.console_logger.info(f'Test epoch end...')
        # final metric calculation for all datasets
        for i, _ in enumerate(self.frame_metric):
            self.compute_metric(stage='test', dataloader_idx=i)
        self.eval_cache.close()
        

def main(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        hparam = ConfigNamespace(**config)
    
    with open(args.assess_config, 'r') as file:
        assess_cfg = yaml.safe_load(file)
        # update args
        for key, value in assess_cfg.items():
            setattr(args, key, value)
   
    logger_root = f'{BASE_EXP_DIR}/{args.exp_name}/test' if args.test_only else f'{BASE_EXP_DIR}/{args.exp_name}/train'
    tblogger = TensorBoardLogger(save_dir=logger_root)
    
    L.seed_everything(args.seed, workers=True)
    
    if args.checkpoint is not None:
        # if some hyperparameters should be overrided, pass it as kwargs
        model = CustomModule.load_from_checkpoint(args.checkpoint, map_location='cpu', args=args, tblogger=tblogger) 
        print(f"Load model CKPT from {args.checkpoint}")
    else:
        model = CustomModule(args, hparam, tblogger)
        print(f"Create model from scratch")
    
    lit_dataset = DataModule(args)
    callbacks = []

    sam_cfg = getattr(getattr(hparam, 'optim', None), 'sam', None)
    if bool(getattr(sam_cfg, 'enabled', False)):
        print("Using SAM optimizer with hyperparameters", sam_cfg)
        callbacks.append(
            SAM(
                rho=sam_cfg.rho,
                adaptive=getattr(sam_cfg, 'adaptive', False),
            )
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tblogger.log_dir, 'checkpoints'),
        filename='{epoch}-{validate_loss:.5f}',
        every_n_epochs=1,
        save_top_k=3,
        monitor='validate_loss',  # validate_loss, validate/{type}/eer
        mode='min',
        save_last=True,
        save_weights_only=True,
        enable_version_counter=True,
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping("validate_loss", patience=args.early_stop)
    callbacks.extend([checkpoint_callback, early_stopping_callback])
   
    trainer = L.Trainer(
        accelerator='gpu',
        devices=args.gpu,
        max_epochs=args.max_epochs,
        logger=[tblogger],
        check_val_every_n_epoch=args.validate_interval,
        callbacks=callbacks
    )
    if args.test_only:
        print('Start testing.')
        trainer.test(model=model, datamodule=lit_dataset, verbose=False)
        print('Test finish.')
    else:
        print('Start training.')
        trainer.fit(model=model, datamodule=lit_dataset)
        print(f"Best checkpoint saved at {checkpoint_callback.best_model_path}, best score: {checkpoint_callback.best_model_score}")
        print('Start testing.')

        best_path = checkpoint_callback.best_model_path
        last_path = checkpoint_callback.last_model_path
        if best_path and os.path.isfile(best_path):
            test_ckpt_path = best_path
            print(f"Test with best checkpoint: {test_ckpt_path}")
        elif last_path and os.path.isfile(last_path):
            test_ckpt_path = last_path
            print(f"Best checkpoint unavailable, fallback to last checkpoint: {test_ckpt_path}")
        else:
            test_ckpt_path = None
            print('No checkpoint file found, test with current in-memory model weights.')

        trainer.test(model=model, datamodule=lit_dataset, ckpt_path=test_ckpt_path, verbose=False)
        print('Test finish.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-grained Voice Spoofing Detection Model')
    parser.add_argument('--exp_name', type=str, default='test', help="experiment name.")
    parser.add_argument('--config', type=str, required=True, help="Model config file path.")
    parser.add_argument('--assess_config', type=str, required=True, help="assess config file path.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to Model checkpoint')
    parser.add_argument('--early_stop', type=int, default=4, help='Early stopping patience')
    parser.add_argument('--validate_interval', type=int, default=2, help="do validate epoch number")
    parser.add_argument('--gpu', default=[0], help="gpu index", type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--max_epochs', type=int, default=12, help='max train epoch.')
    parser.add_argument('--batch_size', type=int, default=8, help='train dataloader batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='train dataloader of num workers')
    parser.add_argument('--fast_eval', action='store_true', default=False, help='Fast evaluation')
    parser.add_argument('--test_only', action='store_true', default=False, help='Do evaluation')

    # For debug & profiler only
    parser.add_argument('--visualize', action='store_true', default=False, help='Enable visualization of embeddings')
    parser.add_argument('--visualize_embedding', default=None, type=str, choices=['h1', 'h2', 'h3'], help='overriding return embedding type')
    args = parser.parse_args()

    if args.num_workers > 1:
        torch.multiprocessing.set_start_method('spawn')  # good solution if worker threads are enabled !!!!

    if args.test_only and not args.checkpoint:
        raise ValueError("In test_only mode, --checkpoint must be provided.")
    main(args)