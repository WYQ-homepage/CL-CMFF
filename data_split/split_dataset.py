"""Prepare only the fixed historical test split from ./dataset raw tiles.

Only Potsdam, Vaihingen and WHU are supported. No random split, validation run,
training preparation, or external output directory is available.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]/'dataset'
TEST_IDS = {
    'Potsdam': ['6_8','2_11','2_12','2_14','3_11','4_15','5_11','6_15'],
    'Vaihingen': ['6','13','23','32','34','37'],
    'WHU': ['NH49E007013','NH49E009023','NH49E004021','NH49E014022','NH49E011017',
            'NH50E006001','NH49E009012','NH49E009014','NH50E009004','NI49E022011',
            'NH50E016002','NI49E024013','NH50E007001','NH50E010001','NH49E012017',
            'NH49E004011','NH50E007004','NH49E010016','NH49E002016','NH50E013001'],
}
MODALITIES = {'Potsdam':['images','ir','dsm','masks'],
              'Vaihingen':['images','dsm','masks'], 'WHU':['images','ir','sar','masks']}
COLORS = [(255,255,255),(0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(0,0,0)]
cv2.setNumThreads(0)


def read(path, flag=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(str(path),flag)
    if image is None:
        raise FileNotFoundError(f'Missing or unreadable raw data: {path}')
    return image


def color_mask(bgr):
    rgb = bgr[:,:,::-1]
    packed = (rgb[:,:,0].astype(np.int32)<<16)|(rgb[:,:,1].astype(np.int32)<<8)|rgb[:,:,2]
    mask = np.full(rgb.shape[:2],255,dtype=np.uint8)
    for class_id,(r,g,b) in enumerate(COLORS):
        mask[packed == ((r<<16)|(g<<8)|b)] = class_id
    return mask


def dsm_path(tile):
    row,col=tile.split('_')
    folder=ROOT/'Potsdam'/'1_DSM_normalisation'
    for name in [f'dsm_potsdam_{tile}_normalized_lastools.jpg',
                 f'dsm_potsdam_{int(row):02d}_{int(col):02d}_normalized_lastools.jpg',
                 f'dsm_potsdam_{tile}_normalized_lastools.tif',
                 f'dsm_potsdam_{int(row):02d}_{int(col):02d}_normalized_lastools.tif']:
        if (folder/name).is_file():
            return folder/name
    raise FileNotFoundError(f'Normalized DSM missing for Potsdam tile {tile}: {folder}')


def load_tile(dataset,tile):
    if dataset=='Potsdam':
        base=ROOT/'Potsdam'
        optical=tifffile.imread(base/'4_Ortho_RGBIR'/f'top_potsdam_{tile}_RGBIR.tif')
        if optical.ndim!=3 or optical.shape[2]!=4:
            raise ValueError(f'Expected RGBIR TIFF for Potsdam {tile}: {optical.shape}')
        image=optical[:,:,:3][:,:,::-1].copy()  # Save via OpenCV as BGR.
        ir=optical[:,:,3]
        dsm=read(dsm_path(tile))
        h,w=image.shape[:2]
        if dsm.shape!=(h,w):
            dsm=cv2.resize(dsm,(w,h),interpolation=cv2.INTER_LINEAR)
        mask=color_mask(read(base/'5_Labels_all_noBoundary'/f'top_potsdam_{tile}_label_noBoundary.tif',cv2.IMREAD_COLOR))
        arrays={'images':image,'ir':ir,'dsm':dsm,'masks':mask}
    elif dataset=='Vaihingen':
        base=ROOT/'Vaihingen'
        image=read(base/'top'/f'top_mosaic_09cm_area{tile}.tif',cv2.IMREAD_COLOR)
        dsm=read(base/'dsm'/f'dsm_09cm_matching_area{tile}.tif').astype(np.float32)
        mask=color_mask(read(base/'gts_eroded_for_participants'/f'top_mosaic_09cm_area{tile}_noBoundary.tif',cv2.IMREAD_COLOR))
        arrays={'images':image,'dsm':dsm,'masks':mask}
    else:
        base=ROOT/'whu-opt-sar'
        optical=read(base/'optical'/f'{tile}.tif')
        if optical.ndim!=3 or optical.shape[2]!=4:
            raise ValueError(f'Expected four optical channels for WHU {tile}: {optical.shape}')
        image,ir=optical[:,:,:3],optical[:,:,3]
        sar=read(base/'sar'/f'{tile}.tif')
        labels=read(base/'lbl'/f'{tile}.tif')
        h,w=image.shape[:2]
        if sar.shape!=(h,w):
            sar=cv2.resize(sar,(w,h),interpolation=cv2.INTER_LINEAR)
        if labels.shape!=(h,w):
            labels=cv2.resize(labels,(w,h),interpolation=cv2.INTER_NEAREST)
        mask=np.full(labels.shape,7,dtype=np.uint8)
        for original,class_id in [(10,0),(20,1),(30,2),(40,3),(50,4),(60,5),(70,6)]:
            mask[labels==original]=class_id
        arrays={'images':image,'ir':ir,'sar':sar,'masks':mask}
    shape=arrays['masks'].shape
    if any(array.shape[:2]!=shape for array in arrays.values()):
        raise ValueError(f'Misaligned raw tile: {dataset}/{tile}')
    return arrays


def grid(length):
    if length<512:
        raise ValueError('Historical test tiles must be at least 512 pixels wide/high')
    points=list(range(0,length-512+1,512))
    if points[-1]!=length-512:
        points.append(length-512)
    return points


def prepare(dataset,check_only=False):
    output=ROOT/f'{dataset}_split'/'test'
    existing=output.exists()
    if check_only and not existing:
        raise FileNotFoundError(f'Prepared test split does not exist: {output}')
    names=[]
    checked=0
    for tile in TEST_IDS[dataset]:
        arrays=load_tile(dataset,tile)
        h,w=arrays['masks'].shape
        prefix=f'potsdam_{tile}' if dataset=='Potsdam' else f'area{tile}' if dataset=='Vaihingen' else f'whu_{tile}'
        patches=[]
        for y in grid(h):
            for x in grid(w):
                if dataset!='WHU' and np.all(arrays['masks'][y:y+512,x:x+512]==255):
                    continue
                patches.append((f'{prefix}_{y}_{x}.tif',y,x))
        # An existing prepared split is checked, never overwritten or extended.
        selected={0,len(patches)//2,len(patches)-1}
        for index,(name,y,x) in enumerate(patches):
            names.append(name)
            for modality in MODALITIES[dataset]:
                target=output/modality/name
                patch=arrays[modality][y:y+512,x:x+512]
                if existing:
                    if not target.is_file():
                        raise FileNotFoundError(f'Incomplete existing test split: {target}')
                    if index in selected:
                        if not np.array_equal(read(target),patch):
                            raise ValueError(f'Prepared test patch differs from raw tile: {target}')
                        checked+=1
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not cv2.imwrite(str(target),patch):
                        raise OSError(f'Failed to save patch: {target}')
        print(f'{dataset}: {tile}: {len(patches)} test patches',flush=True)
    if existing:
        for modality in MODALITIES[dataset]:
            actual=sorted(p.name for p in (output/modality).glob('*.tif'))
            if actual!=sorted(names):
                raise ValueError(f'Unexpected/missing test samples in {output/modality}')
    info={'dataset':dataset,'test_tile_ids':TEST_IDS[dataset],'samples':len(names),
          'source':'fixed historical tile IDs','crop_size':512,'stride':512,'edge_window':'append last full window',
          'existing_split':'checked without writes' if existing else 'created',
          'pixel_checked_modality_files':checked}
    print(json.dumps(info),flush=True)
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=list(TEST_IDS),default=list(TEST_IDS))
    parser.add_argument('--check',action='store_true',help='Check existing test splits against raw tiles without writing datasets')
    args=parser.parse_args()
    results=[prepare(dataset,args.check) for dataset in args.datasets]
    report=Path(__file__).resolve().parents[1]/'results'/'split_check.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
