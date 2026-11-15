#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlx==0.32.4", "numpy==2.3.5"]
# ///
"""Independent packing, rotation, and int32-rescaling oracles for the M5 kernels."""
import unittest
import mlx.core as mx
import numpy as np
from profile_sov_metal_quant import DynamicInt8


def unpack(packed, block=256):
    q,s,e=[np.array(x) for x in packed]
    rows,k=q.shape
    q=q.reshape(rows//64,k//block,64,block).transpose(0,2,1,3).reshape(rows,k)
    return q,s,e


def hadamard(x):
    a=x.copy().reshape(-1,256)
    distance=1
    while distance<256:
        b=a.reshape(-1,256//(2*distance),2,distance)
        left,right=b[:,:,0,:].copy(),b[:,:,1,:].copy()
        b[:,:,0,:]=left+right;b[:,:,1,:]=left-right
        distance*=2
    return a.reshape(x.shape)/16


class MetalQuantTests(unittest.TestCase):
    def test_fused_packing_matches_independent_cpu_rotation_and_rounding(self):
        rng=np.random.default_rng(17)
        x=np.array(mx.array(rng.normal(size=(64,1280)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
        x[0]=0
        for rotate in [False,True]:
            op=DynamicInt8(256,2,True,True,rotate,True)
            q,s,e=unpack(op.pack(mx.array(x)))
            rotated=hadamard(x) if rotate else x
            expected_scale=np.maximum(np.max(np.abs(rotated),axis=1)/127,1e-12)
            maxima=np.max(np.abs(rotated.reshape(64,5,256)),axis=-1)
            ratio=127*expected_scale[:,None]/np.maximum(maxima,1e-30)
            expected_exp=np.clip(np.floor(np.log2(ratio)),0,2).astype(np.uint8)
            expected_exp[maxima==0]=0
            expected_q=np.clip(np.rint(rotated.reshape(64,5,256)*(np.exp2(expected_exp.astype(np.float32))/expected_scale[:,None])[:,:,None]),-127,127).astype(np.int8).reshape(64,1280)
            np.testing.assert_allclose(s,expected_scale,rtol=2e-7,atol=1e-12)
            np.testing.assert_array_equal(e,expected_exp)
            delta=np.abs(q.astype(np.int16)-expected_q.astype(np.int16))
            self.assertLessEqual(delta.max(),1)
            ideal=rotated.reshape(64,5,256)*(np.exp2(expected_exp.astype(np.float32))/expected_scale[:,None])[:,:,None]
            ties=np.abs(np.abs(ideal)-np.floor(np.abs(ideal))-0.5)<3e-5
            self.assertTrue(np.all((delta==0)|ties.reshape(64,1280)))

    def test_tensorops_matches_integer_oracle_with_alternating_exponents(self):
        rng=np.random.default_rng(21)
        for rotate,exponent in [(False,0),(True,0),(False,2),(True,2)]:
            op=DynamicInt8(256,exponent,True,True,rotate,True)
            x=rng.normal(size=(128,1280)).astype(np.float32)
            w=rng.normal(size=(64,1280)).astype(np.float32)
            x=x.reshape(128,5,256)*np.array([1,0.25,0.5,1,0.25])[None,:,None]
            w=w.reshape(64,5,256)*np.array([0.25,1,0.25,0.5,1])[None,:,None]
            a=mx.array(x.reshape(128,1280)).astype(mx.bfloat16)
            b=mx.array(w.reshape(64,1280)).astype(mx.bfloat16)
            aq,asc,ae=unpack(op.pack(a));packed=op.pack(b);bq,bsc,be=unpack(packed)
            acc=np.zeros((128,64),dtype=np.int64);previous=np.zeros_like(acc)
            for block in range(5):
                exp=ae[:,block,None].astype(np.int64)+be[None,:,block].astype(np.int64)
                for change in range(-4,5):
                    mask=(exp-previous)==change
                    if change>0:acc[mask]<<=change
                    if change<0:acc[mask]=(acc[mask]+(1<<(-change-1)))>>-change
                acc+=aq[:,block*256:(block+1)*256].astype(np.int64)@bq[:,block*256:(block+1)*256].astype(np.int64).T
                previous=exp
            oracle=(acc.astype(np.float32)*np.exp2(-previous.astype(np.float32))*asc[:,None]*bsc[None,:])
            actual=op.matmul(a[:67],packed,mx.zeros((64,),dtype=mx.bfloat16))
            # One BF16 ULP allows final F32 multiply rounding at conversion boundaries.
            out=np.array(actual.astype(mx.float32));expected=np.array(mx.array(oracle[:67]).astype(mx.bfloat16).astype(mx.float32))
            ulp=np.maximum(np.abs(np.spacing(expected.astype(np.float32)))*65536,1e-8)
            self.assertTrue(np.all(np.abs(out-expected)<=ulp))
            self.assertLess(np.linalg.norm(out-expected)/np.linalg.norm(expected),1e-4)

    def test_invalid_geometry_is_rejected_before_gpu_dispatch(self):
        op=DynamicInt8(64,2,True,True,False,True)
        with self.assertRaises(ValueError):op.pack(mx.zeros((64,5120)))
        with self.assertRaises(ValueError):op.pack(mx.zeros((63,256)))
        with self.assertRaises(ValueError):op.pack(mx.zeros((64,257)))


if __name__=='__main__':unittest.main()
