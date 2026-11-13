"""Test-only TIFF loader using the supplied training-set normalization statistics."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)


def read(path, flag=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f'Unreadable image: {path}')
    return image


def load_raw(root, dataset, split, name):
    base = Path(root)/f'{dataset}_split'/split
    rgb = cv2.cvtColor(read(base/'images'/name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = read(base/'masks'/name, cv2.IMREAD_GRAYSCALE)
    modalities = ['dsm'] if dataset == 'Vaihingen' else ['ir', 'dsm' if dataset == 'Potsdam' else 'sar']
    auxiliary = [read(base/m/name) for m in modalities]
    if any(a.shape != mask.shape for a in auxiliary) or rgb.shape[:2] != mask.shape:
        raise ValueError(f'Misaligned sample: {dataset}/{split}/{name}')
    return rgb, auxiliary, mask


def filenames(root, dataset, split):
    names = sorted(p.name for p in (Path(root)/f'{dataset}_split'/split/'images').glob('*.tif'))
    if not names:
        raise ValueError(f'No TIFF samples: {dataset}/{split}')
    return names



class EvaluationDataset(Dataset):
    def __init__(self, root, dataset, split, stats, limit=None):
        self.root, self.dataset, self.split, self.stats = Path(root), dataset, split, stats
        self.names = filenames(root,dataset,split)
        if limit:
            self.names = self.names[:limit]

    def __len__(self):
        return len(self.names)

    def __getitem__(self,index):
        rgb, aux, mask = load_raw(self.root,self.dataset,self.split,self.names[index])
        stats = self.stats
        rm, rs = np.array(stats['rgb_mean'],dtype=np.float32), np.array(stats['rgb_std'],dtype=np.float32)
        xm, xs = np.array(stats['x_mean'],dtype=np.float32), np.array(stats['x_std'],dtype=np.float32)
        ignore = mask == 255
        if self.dataset == 'Potsdam':
            x = np.dstack([aux[0], aux[1].astype(np.float32)])
            rgb[ignore] = (rm*255).astype(np.uint8)
            x[ignore] = xm*np.array([255.,1.],dtype=np.float32)
            x[:,:,0] /= 255.
        elif self.dataset == 'Vaihingen':
            x = aux[0].astype(np.float32)[...,None]
        else:
            # Legacy checkpoint preprocessing always scales WHU IR/SAR by 255.
            x = np.dstack([a.astype(np.float32)/255. for a in aux])
            mask = mask.copy()
            mask[mask>=8] = 255
        rgb = (rgb.astype(np.float32)/255.-rm)/(rs+1e-6)
        if self.dataset == 'Vaihingen':
            # Preserve scalar arithmetic from the historical DSM loader.
            x = (x-stats['x_mean'][0])/(stats['x_std'][0]+1e-6)
        else:
            x = (x-xm)/(xs+1e-6)
        if self.dataset == 'Vaihingen':
            rgb[ignore] = 0.; x[ignore] = 0.
            x = np.clip(x,-5.,5.)
        return torch.from_numpy(rgb.transpose(2,0,1)).float(), torch.from_numpy(x.transpose(2,0,1)).float(), torch.from_numpy(mask.astype(np.int64))
