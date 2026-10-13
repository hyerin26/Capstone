import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_attention import summarize_maps, parse_side

class GroundingMetrics(unittest.TestCase):
    def test_uniform_attention_area_correction(self):
        maps=np.ones((2,3,2,4),dtype='float32')/8
        left=np.zeros((2,4),dtype=bool);left[:,:1]=True
        right=np.zeros((2,4),dtype=bool);right[:,1:]=True
        m=summarize_maps(maps,dict(left=left,right=right))
        np.testing.assert_allclose(m['left_fraction_by_layer'],[.25,.25])
        np.testing.assert_allclose(m['right_fraction_by_layer'],[.75,.75])
        np.testing.assert_allclose(m['left_density_by_layer'],m['right_density_by_layer'])
    def test_swapping_spatial_attention_swaps_scores(self):
        maps=np.zeros((1,2,2,4),dtype='float32');maps[...,0]=.5
        left=np.zeros((2,4),dtype=bool);left[:,:2]=True
        masks=dict(left=left,right=~left)
        a=summarize_maps(maps,masks);b=summarize_maps(maps[...,::-1],masks)
        np.testing.assert_allclose(a['left_fraction_by_layer'],b['right_fraction_by_layer'])
        self.assertTrue(a['left_peak_hit_by_layer'][0])
        self.assertFalse(a['right_peak_hit_by_layer'][0])
    def test_strict_response_parser(self):
        self.assertEqual(parse_side(' RIGHT. '),'right')
        self.assertIsNone(parse_side('LEFT or RIGHT'))
        self.assertIsNone(parse_side('The answer is left'))

if __name__=='__main__':unittest.main()
