"""Infer once per sample, then calculate wrong and correct metrics."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_process.dataset import EvaluationDataset
from utils.metrics import CLASS_NAMES, SUMMARY_KEYS, save_comparison


HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(dataset, args):
    from model.core import Model
    import cv2
    import timm
    import torchvision
    print(f'\n{dataset}: preparing {args.split} evaluation', flush=True)
    stats_path = HERE/'configs'/f'{dataset}.json'
    if not stats_path.is_file():
        raise FileNotFoundError(f'Missing supplied normalization statistics: {stats_path}')
    stats = json.loads(stats_path.read_text())
    ds = EvaluationDataset(args.data_root,dataset,args.split,stats,args.limit)
    names_manifest = '\n'.join(ds.names)+'\n'
    out = args.output/args.split/dataset
    out.mkdir(parents=True,exist_ok=True)
    (out/'samples.txt').write_text(names_manifest,encoding='utf-8')
    loader = DataLoader(ds,batch_size=args.batch_size,num_workers=args.workers,shuffle=False,drop_last=False,pin_memory=args.device.startswith('cuda'))
    num_classes = len(CLASS_NAMES[dataset])
    checkpoint = args.weights/f'{dataset}.pth'
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if isinstance(state,dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    model = Model(num_classes=num_classes,light_in_chans=1 if dataset=='Vaihingen' else 2,model_use='all')
    model.load_state_dict(state,strict=True)
    del state
    model.to(args.device).eval()
    cm = torch.zeros((num_classes,num_classes),dtype=torch.int64,device=args.device)
    ignored_pixels = 0
    started = time.time()
    # FP32, no TTA or mixed precision: the same predictions feed both protocols.
    with torch.inference_mode():
        for i,(rgb,x,labels) in enumerate(loader):
            labels = labels.to(args.device)
            logits = model(rgb.to(args.device,non_blocking=True),x.to(args.device,non_blocking=True))
            if logits.shape[1] != num_classes or logits.shape[2:] != labels.shape[1:]:
                raise ValueError(f'Unexpected logits shape: {logits.shape}')
            pred = logits.argmax(1)
            valid = (labels>=0)&(labels<num_classes)
            ignored_pixels += int((~valid).sum().item())
            indices = num_classes*labels[valid]+pred[valid]
            cm += torch.bincount(indices,minlength=num_classes*num_classes).reshape(num_classes,num_classes)
            if (i+1)%25==0 or i+1==len(loader):
                print(f'{dataset}: {min((i+1)*args.batch_size,len(ds))}/{len(ds)} samples, {time.time()-started:.1f}s',flush=True)
    cm = cm.cpu().numpy()
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = None
    provenance = {'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
                  'split':args.split,'samples':len(ds),'partial_run':args.limit is not None,
                  'ignored_pixels':ignored_pixels,'data_root':str(args.data_root.resolve()),
                  'sample_list_sha256':hashlib.sha256(names_manifest.encode()).hexdigest(),
                  'stats':stats,'stats_sha256':sha256(stats_path),
                  'model_mode':'all','precision':'float32','tta':False,
                  'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,
                  'numpy':np.__version__,'opencv':cv2.__version__,'timm':timm.__version__,
                  'device':args.device,'seconds':time.time()-started,'commit':commit,
                  'source_sha256':{str(p.relative_to(HERE)):sha256(p) for p in sorted(HERE.rglob('*.py'))}}
    result = save_comparison(cm,dataset,out,provenance)
    # Retain original evaluator numerical output for exact historical checking.
    from utils.legacy_metrics import calculate_legacy
    legacy = calculate_legacy(cm,num_classes-1)
    (out/'legacy_float32_summary.json').write_text(json.dumps(legacy,indent=2)+'\n',encoding='utf-8')
    for key in ['miou','mf1','oa','mean_recall']:
        if abs(legacy[key]-result['wrong']['summary'][key]) > .001:
            raise AssertionError(f'Wrong protocol does not match legacy evaluator: {key}')
    del model,loader
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def summary(results,output):
    rows = []
    for result in results:
        row = {'dataset':result['dataset']}
        for key in SUMMARY_KEYS:
            for protocol in ['wrong','correct']:
                row[f'{protocol}_{key}'] = result[protocol]['summary'][key]
            row[f'delta_{key}_pp'] = result['delta_correct_minus_wrong_pp'][key]
        row['discarded_pixels_wrong'] = result['wrong']['summary']['discarded_pixels']
        rows.append(row)
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/'all_results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = ['# Measured differences', '', 'All values are percentages; delta = correct − wrong, in percentage points.', '',
             '| Dataset | Wrong mIoU | Correct mIoU | ΔmIoU | Wrong mF1 | Correct mF1 | ΔmF1 | Wrong OA | Correct OA | ΔOA |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        keys = ['wrong_miou','correct_miou','delta_miou_pp','wrong_mf1','correct_mf1','delta_mf1_pp','wrong_oa','correct_oa','delta_oa_pp']
        lines.append('| '+row['dataset']+' | '+' | '.join(f'{row[k]:.4f}' for k in keys)+' |')
    lines += ['', '| Dataset | Correct mean Recall | Foreground → excluded | Excluded → foreground | Pixels discarded by wrong method |',
              '|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["dataset"]} | {r["correct"]["summary"]["mean_recall"]:.4f} | {r["foreground_to_excluded_pixels"]} | {r["excluded_to_foreground_pixels"]} | {r["wrong"]["summary"]["discarded_pixels"]} |')
    lines += ['', 'Per-class IoU/F1/Precision/Recall, TP/FP/FN, full matrices and manifests are in each dataset directory.',
              'Clutter/Background is omitted from the per-class tables and macro averages.',
              'This comparison holds checkpoint, samples, preprocessing, predictions and patch coverage fixed.',
              'The correction changes metric calculation; it does not retroactively correct historical checkpoint selection.', '']
    (output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=list(CLASS_NAMES),default=list(CLASS_NAMES))
    parser.add_argument('--data-root',type=Path,default=HERE/'dataset')
    parser.add_argument('--weights',type=Path,default=HERE/'weights')
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,help='Smoke test only; outputs are marked partial')
    args = parser.parse_args()
    args.split = 'test'
    if args.batch_size < 1 or args.workers < 0 or (args.limit is not None and args.limit < 1):
        parser.error('Invalid batch size, workers or sample limit')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True,exist_ok=True)
    results = [evaluate(dataset,args) for dataset in args.datasets]
    print(summary(results,args.output/args.split),flush=True)


if __name__ == '__main__':
    main()
