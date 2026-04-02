from typing import Any, Dict, Iterator, List, Optional, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _get_params(optimizer: Optimizer) -> Iterator[Tensor]:
    for param_group in cast(List[Dict[Any, Any]], optimizer.param_groups):
        for param in param_group["params"]:
            if not isinstance(param, Tensor):
                raise TypeError(f"expected Tensor, but got: {type(param)}")
            yield param


def _get_loss(step_output: STEP_OUTPUT) -> Optional[Tensor]:
    if step_output is None:
        return None
    if isinstance(step_output, Tensor):
        return step_output
    return step_output.get("loss")


class SAM(Callback):
    def __init__(self, rho: float = 0.05, adaptive: bool = False) -> None:
        super().__init__()
        self._rho = rho
        self._adaptive = adaptive
        self._batch: Any = None
        self._batch_idx: int = 0

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._batch = batch
        self._batch_idx = batch_idx

    @torch.no_grad()
    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self._batch is None:
            return

        org_weights = self._first_step(optimizer)
        if not org_weights:
            self._batch = None
            return
        with torch.enable_grad():
            if hasattr(pl_module, "compute_objective"):
                loss = pl_module.compute_objective(
                    self._batch,
                    stage='train',
                    update_metric=False,
                    log_detail=False,
                )
            else:
                step_output = pl_module.training_step(self._batch, self._batch_idx)
                loss = _get_loss(step_output)

            if loss is not None:
                trainer.strategy.backward(loss, optimizer=optimizer)
        self._second_step(optimizer, org_weights)
        self._batch = None

    def _norm_weights(self, parameter: Tensor) -> Tensor:
        return torch.abs(parameter) if self._adaptive else torch.ones_like(parameter)

    def _grad_norm(self, optimizer: Optimizer) -> Tensor:
        param_norm_list = [
            (self._norm_weights(parameter) * parameter.grad).norm()
            for parameter in _get_params(optimizer)
            if isinstance(parameter.grad, Tensor)
        ]
        if not param_norm_list:
            return torch.tensor(0.0)
        param_norms = torch.stack(param_norm_list)
        return param_norms.norm()

    def _first_step(self, optimizer: Optimizer) -> Dict[Tensor, Tensor]:
        has_grad = any(parameter.grad is not None for parameter in _get_params(optimizer))
        if not has_grad:
            return {}
        scale = self._rho / (self._grad_norm(optimizer) + 1e-4)
        org_weights: Dict[Tensor, Tensor] = {}
        for parameter in _get_params(optimizer):
            if parameter.grad is None:
                continue
            org_weights[parameter] = parameter.data.clone()
            e_w = (torch.pow(parameter, 2) if self._adaptive else 1.0) * parameter.grad * scale.to(parameter)
            parameter.add_(e_w)
        optimizer.zero_grad()
        return org_weights

    def _second_step(self, optimizer: Optimizer, org_weights: Dict[Tensor, Tensor]) -> None:
        for parameter in _get_params(optimizer):
            if parameter.grad is None:
                continue
            parameter.data = org_weights[parameter]
