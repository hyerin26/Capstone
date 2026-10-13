"""Deterministic experimental image transformations; never alter source scenes."""
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
from PIL import Image, ImageDraw, ImageFont


def quadrilateral(mask):
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=max(contours,key=cv2.contourArea);hull=cv2.convexHull(contour)
    quad=None
    for eps in np.linspace(.005,.08,76):
        candidate=cv2.approxPolyDP(hull,eps*cv2.arcLength(hull,True),True)
        if len(candidate)==4:quad=candidate[:,0].astype('float32');break
    if quad is None:raise ValueError('No four-corner tile approximation')
    # In this fixed camera, the rear-left board corner is the topmost vertex.
    # Clockwise image-coordinate order preserves the printed image orientation.
    center=quad.mean(0)
    quad=quad[np.argsort(np.arctan2(quad[:,1]-center[1],quad[:,0]-center[0]))]
    quad=np.roll(quad,-int(quad[:,1].argmin()),axis=0)
    if not (quad[1,0]>quad[0,0] and quad[2,1]>quad[1,1]):
        raise ValueError('Unvalidated fixed-camera corner orientation')
    polygon=np.zeros(mask.shape,dtype='uint8');cv2.fillConvexPoly(polygon,quad.astype('int32'),1)
    iou=float((polygon.astype(bool)&mask).sum()/(polygon.astype(bool)|mask).sum())
    if iou<.90:raise ValueError(f'Poor quadrilateral overlap: {iou}')
    return quad,iou


def marked(image,masks,mapping):
    im=image.copy();draw=ImageDraw.Draw(im)
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans-Bold.ttf'),24)
    for side,label in zip(('left','right'),mapping):
        ys,xs=np.nonzero(masks[side]);cx=int(round((xs.min()+xs.max())/2))
        top=int(ys.min())-36
        if top<0:raise ValueError('Insufficient label margin')
        draw.rectangle((cx-18,top,cx+18,top+30),fill='white',outline='black',width=2)
        draw.text((cx,top+15),label,fill='black',font=font,anchor='mm')
    return im


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenes',default='outputs/scenes');p.add_argument('--output',default='outputs/followup_inputs');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    records=[];preview={}
    for f in sorted(Path(a.scenes).glob('*/metadata.json')):
        meta=json.loads(f.read_text());image=Image.open(f.parent/'rgb.png').convert('RGB');assert image.size==(640,480)
        source=np.array(image);masks={s:np.array(Image.open(f.parent/f'{s}_mask.png'))>0 for s in ('left','right')}
        target=out/meta['scene_id'];target.mkdir(exist_ok=True)
        canvas=Image.new('RGB',image.size,(127,127,127));equal_masks={};geometry={}
        for side,x in [('left',32),('right',352)]:
            quad,iou=quadrilateral(masks[side]);dest=np.array([[0,0],[255,0],[255,255],[0,255]],dtype='float32')
            transform=cv2.getPerspectiveTransform(quad,dest)
            tile=cv2.warpPerspective(source,transform,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(127,127,127))
            canvas.paste(Image.fromarray(tile),(x,112));mask=np.zeros((480,640),bool);mask[112:368,x:x+256]=True;equal_masks[side]=mask
            geometry[side]=dict(corners=quad.tolist(),mask_quad_iou=iou,original_mask_pixels=int(masks[side].sum()),homography=transform.tolist())
        images={'original':image,'equal':canvas}
        for layout,im,ms in [('original',image,masks),('equal',canvas,equal_masks)]:
            for mapping in ('AB','BA'):images[f'{layout}_{mapping}']=marked(im,ms,mapping)
        hashes={}
        for name,im in images.items():
            path=target/f'{name}.png';im.save(path);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        record=dict(meta,geometry=geometry,image_sha256=hashes,original_rgb_sha256=meta['rgb_sha256'])
        (target/'metadata.json').write_text(json.dumps(record,indent=2));records.append(record)
        preview.setdefault(meta['category'],(target,meta))
    (out/'manifest.json').write_text(json.dumps(dict(cases=len(records),min_iou=min(r['geometry'][s]['mask_quad_iou'] for r in records for s in ('left','right')),source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),17)
    for category,(folder,meta) in preview.items():
        panel=Image.new('RGB',(1280,1020),'white');draw=ImageDraw.Draw(panel)
        for i,name in enumerate(('original','equal','original_AB','equal_AB')):
            x=(i%2)*640;y=(i//2)*510
            draw.text((x+8,y+5),f'{category}: {name}',font=font,fill='black');panel.paste(Image.open(folder/f'{name}.png'),(x,y+30))
        panel.save(out/f'preview_{category}.png')
    print((out/'manifest.json').read_text())

if __name__=='__main__':main()
