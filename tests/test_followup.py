import importlib.util, math, sys, unittest
from pathlib import Path
import numpy as np
import cv2
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'/'followup'
def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
run=load('run');prep=load('prepare_inputs')
class FollowupChecks(unittest.TestCase):
    def test_spelling_marginal_can_correct_uppercase_ranking(self):
        probs=dict(LEFT=.10,RIGHT=.20,Left=.60,Right=.05,left=.025,right=.025)
        scores=run.score_sides({k:math.log(v) for k,v in probs.items()},'lr','')
        self.assertGreater(scores['uppercase']['right'],scores['uppercase']['left'])
        self.assertGreater(scores['case_marginal']['left'],scores['case_marginal']['right'])
        self.assertAlmostEqual(math.exp(scores['case_marginal']['left']),.725)
    def test_label_mapping_reversal(self):
        self.assertEqual(run.parse_response('A','ab','AB'),'left')
        self.assertEqual(run.parse_response('A','ab','BA'),'right')
        self.assertEqual(run.parse_response('b.','ab','BA'),'left')
        self.assertIsNone(run.parse_response('A or B','ab','AB'))
    def test_perspective_corner_order_preserves_orientation(self):
        corners=np.array([[195,91],[297,139],[229,210],[103,159]],dtype='float32')
        mask=np.zeros((480,640),dtype='uint8');cv2.fillConvexPoly(mask,corners.astype('int32'),1)
        q,iou=prep.quadrilateral(mask.astype(bool))
        np.testing.assert_allclose(q,corners,atol=1)
        self.assertGreater(iou,.99)
        dest=np.array([[0,0],[255,0],[255,255],[0,255]],dtype='float32')
        transformed=cv2.perspectiveTransform(corners[None],cv2.getPerspectiveTransform(q,dest))[0]
        np.testing.assert_allclose(transformed,dest,atol=2)
if __name__=='__main__':unittest.main()
