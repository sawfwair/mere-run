"""Deterministic CPU 3DGS inspection renderer for qualification (not a product renderer).

Projects anisotropic covariance, depth-sorts splats, and composites degree-0 SH
colour. Requires numpy, numba, pillow, plyfile. Both inputs use the same camera
and pixel filter; comparisons require matched noise, latent and anchors.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from numba import njit
from PIL import Image
from plyfile import PlyData

@njit
def composite(xy,cov,z,colours,opacity,size):
    image=np.ones((size,size,3),np.float32)*.92
    coverage=np.zeros((size,size),np.float32)
    for index in np.argsort(z)[::-1]:
        if z[index]<=.01:continue
        a,b,c=cov[index,0,0],cov[index,0,1],cov[index,1,1]
        determinant=a*c-b*b
        if determinant<=0:continue
        radius=min(size,int(math.ceil(3*math.sqrt(max(a,c)+abs(b)))))
        px,py=xy[index,0],xy[index,1]
        for y in range(max(0,int(py)-radius),min(size,int(py)+radius+1)):
            for x in range(max(0,int(px)-radius),min(size,int(px)+radius+1)):
                dx,dy=x+.5-px,y+.5-py
                power=-.5*(c*dx*dx-2*b*dx*dy+a*dy*dy)/determinant
                if power>0 or power < -9:continue
                alpha=min(.99,opacity[index]*math.exp(power))
                if alpha<1/255:continue
                for channel in range(3):image[y,x,channel]=colours[index,channel]*alpha+image[y,x,channel]*(1-alpha)
                coverage[y,x]=alpha+coverage[y,x]*(1-alpha)
    return image,coverage

def load(path):
    data=PlyData.read(path)['vertex'].data
    def cols(names):return np.stack([data[name] for name in names],axis=1).astype(np.float32)
    xyz=cols(['x','y','z']);scale=np.exp(cols(['scale_0','scale_1','scale_2']))
    quaternion=cols(['rot_0','rot_1','rot_2','rot_3']);quaternion/=np.maximum(np.linalg.norm(quaternion,axis=1,keepdims=True),1e-12)
    w,x,y,z=quaternion.T
    rotation=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                       2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=1).reshape(-1,3,3)
    covariance=(rotation*(scale*scale)[:,None,:])@rotation.transpose(0,2,1)
    colours=np.clip(.5+.28209479177387814*cols(['f_dc_0','f_dc_1','f_dc_2']),0,1)
    opacity=1/(1+np.exp(-np.clip(data['opacity'],-80,80)))
    assert all(np.isfinite(v).all() for v in [xyz,scale,quaternion,colours,opacity])
    return xyz,covariance,colours,opacity

def render(model,angle,size=256):
    xyz,cov,colours,opacity=model
    yaw=math.radians(angle);eye=np.array([2.3*math.sin(yaw),.5,2.3*math.cos(yaw)],np.float32)
    forward=-eye/np.linalg.norm(eye);right=np.cross(forward,np.array([0,1,0],np.float32));right/=np.linalg.norm(right)
    up=np.cross(right,forward);camera=np.stack([right,up,forward])
    points=(xyz-eye)@camera.T;x,y,z=points.T;focal=size*.5/math.tan(math.radians(35)*.5)
    xy=np.stack([focal*x/z+size/2,-focal*y/z+size/2],axis=1)
    jacobian=np.zeros((len(xyz),2,3),np.float32)
    jacobian[:,0,0]=focal/z;jacobian[:,0,2]=-focal*x/(z*z)
    jacobian[:,1,1]=-focal/z;jacobian[:,1,2]=focal*y/(z*z)
    cam_cov=camera[None]@cov@camera.T[None]
    projected=jacobian@cam_cov@jacobian.transpose(0,2,1)
    projected[:,0,0]+=.3;projected[:,1,1]+=.3
    return composite(xy,projected,z,colours,opacity,size)

p=argparse.ArgumentParser();p.add_argument('--ply',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--reference',type=Path);a=p.parse_args()
model=load(a.ply);baseline=load(a.reference) if a.reference else None
images=[];metrics={}
for angle in [0,120,240]:
    image,coverage=render(model,angle);images.append(image)
    if baseline:
        expected,alpha=render(baseline,angle)
        mse=np.mean((image-expected)**2);mask=coverage>.5;expected_mask=alpha>.5
        metrics[str(angle)]={'psnr_db':float(-10*np.log10(max(float(mse),1e-20))),
            'foreground_iou':float(np.sum(mask&expected_mask)/max(np.sum(mask|expected_mask),1))}
contact=np.concatenate(images,axis=1)
a.output.parent.mkdir(parents=True,exist_ok=True)
Image.fromarray(np.clip(contact*255,0,255).round().astype(np.uint8)).save(a.output)
metrics['splats']=len(model[0]);metrics['bounds']=[model[0].min(axis=0).tolist(),model[0].max(axis=0).tolist()]
a.output.with_suffix('.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
