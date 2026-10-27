#!/usr/bin/env python3
"""Protect affine packing, budget allocation, and calibration/heldout separation."""
import unittest
from pathlib import Path
import numpy as np
from lfm25_quality18 import allocation,pack_codes
from build_lfm25_quality_v2_suite import build
from compare_lfm25_native import metrics

class Quality18Tests(unittest.TestCase):
    def test_all_precisions_roundtrip_across_word_and_row_boundaries(self):
        for bits in [4,5,6,8]:
            codes=np.random.default_rng(bits).integers(0,2**bits,size=(3,7,128),dtype=np.uint8)
            packed=pack_codes(codes,bits)
            actual=np.empty_like(codes)
            for column in range(codes.shape[-1]):
                word,shift=divmod(column*bits,32)
                value=packed[...,word].astype(np.uint64)>>shift
                if shift+bits>32:value|=packed[...,word+1].astype(np.uint64)<<(32-shift)
                actual[...,column]=(value&(2**bits-1)).astype(np.uint8)
            np.testing.assert_array_equal(actual,codes)
            self.assertEqual(packed.shape,(3,7,4*bits))
    def test_packing_rejects_partial_blocks_and_unknown_precision(self):
        with self.assertRaises(ValueError):pack_codes(np.zeros((3,33),dtype=np.uint8),5)
        with self.assertRaises(ValueError):pack_codes(np.zeros((3,64),dtype=np.uint8),7)
    def test_allocator_respects_budget_and_ignores_disallowed_options(self):
        banks={str(i):{'low':dict(logical_bytes=100,relative_block_output_mse=.1*(i+1)),
                       'high':dict(logical_bytes=150,relative_block_output_mse=.01),
                       'excluded':dict(logical_bytes=100,relative_block_output_mse=0.)} for i in range(4)}
        policy,total=allocation(banks,{'low','high'},560,10)
        self.assertLessEqual(total,560)
        self.assertEqual(sum(v=='high' for v in policy.values()),3)
        self.assertEqual(policy['3'],'high')
        self.assertNotIn('excluded',policy.values())
    def test_reference_metrics_match_closed_form_and_constant_shift(self):
        x=np.log(np.array([[.75,.25],[.25,.75]]));y=np.log(np.full((2,2),.5));target=np.array([0,1])
        shifted=metrics(x,x+42,target);changed=metrics(x,y,target)
        self.assertAlmostEqual(shifted['mean_kl'],0.,places=13)
        self.assertAlmostEqual(shifted['perplexity_ratio'],1.,places=13)
        self.assertAlmostEqual(changed['mean_kl'],.75*np.log(1.5)+.25*np.log(.5),places=13)
        self.assertAlmostEqual(changed['perplexity_ratio'],1.5,places=13)
    def test_authored_calibration_and_heldout_prompts_are_disjoint(self):
        suite=build(Path(__file__).with_name('lfm25-24b-quality-suite.json'))
        rows=suite['cases'];self.assertEqual(len(rows),220)
        ids=[r['id'] for r in rows];self.assertEqual(len(ids),len(set(ids)))
        prompts={split:{r['messages'][0]['content'] for r in rows if r['split']==split} for split in ['calibration','heldout']}
        self.assertFalse(prompts['calibration']&prompts['heldout'])
        self.assertEqual(suite['maximum_logical_weight_bytes'],19_000_000_000)

if __name__=='__main__':unittest.main()
