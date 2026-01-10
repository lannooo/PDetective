import os
import fnmatch
import torch
import numpy as np
import importlib
import importlib.util

def import_class_from_path(target_class, target_path):
    cls_name = target_class.split('.')[-1]
    spec = importlib.util.spec_from_file_location('model', target_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package_path = module.__file__
    cls = getattr(module,cls_name)
    return cls, package_path

def import_class(target_class):
    cla_seps = target_class.split('.')
    module = importlib.import_module('.'.join(cla_seps[:-1]))
    cls = getattr(module, cla_seps[-1])
    package_path = module.__file__
    return cls, package_path


def get_boundary_label(label):
    pos = []
    for i, l in enumerate(label):
        if i == 0:
            last = l
        if l != last:
            splice_index = i if l == 0 else i - 1
            pos.append(splice_index)
            last = l
    pos = list(set(pos))
    boundary_label= np.zeros_like(label)
    boundary_label[pos] = 1.0
    return boundary_label


def get_mask(label, length=None):
    mask = torch.ones_like(label)
    if len(mask.size()) > 2:
        mask = mask[:, :, 0]
    if length is None:
        return mask
    for i, l in enumerate(length):
        mask[i][int(l):] = 0.
    return mask

def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files
