import sys
import os
import unittest
from types import SimpleNamespace
import numpy as np
from evaluate import prepare, iou
from controller import StableCalibrator
if os.environ.get('TRACKEVAL_PATH'):
    sys.path.insert(0, os.environ['TRACKEVAL_PATH'])
import trackeval

class EvaluationTests(unittest.TestCase):
    def data(self, ids):
        gt = [[f, 1, 0, 0, 10, 10, 1, 1, 0, 0] for f in range(1, len(ids)+1)]
        frames = [dict(frame=f, ids=[pid], boxes_xyxy=[[0,0,10,10]]) for f,pid in enumerate(ids,1)]
        return prepare(gt, frames, len(ids))

    def test_perfect_and_switch(self):
        h = trackeval.metrics.HOTA()
        c = trackeval.metrics.CLEAR({'PRINT_CONFIG':False})
        self.assertAlmostEqual(h.eval_sequence(self.data([7,7]))['HOTA'].mean(), 1)
        self.assertLess(h.eval_sequence(self.data([7,8]))['HOTA'].mean(), 1)
        self.assertEqual(c.eval_sequence(self.data([7,8]))['IDSW'], 1)

    def test_no_overlap(self):
        self.assertEqual(iou([[0,0,10,10]], [[20,20,30,30]])[0,0], 0)

    def test_missing_frame(self):
        with self.assertRaises(ValueError):
            prepare([], [dict(frame=2,ids=[],boxes_xyxy=[])], 2)

    def test_duplicates(self):
        with self.assertRaises(ValueError):
            prepare([], [dict(frame=1,ids=[1,1],boxes_xyxy=[[0,0,1,1]]*2)], 1)

    def test_resolution_dwell(self):
        c = StableCalibrator()
        state = SimpleNamespace(sci=.7,tiny_ratio=.7,scene='tiny')
        for _ in range(30):
            self.assertEqual(c.params(state)['imgsz'],640)
        c.params(SimpleNamespace(**vars(state)))
        self.assertEqual(c.params(SimpleNamespace(**vars(state)))['imgsz'],832)
        self.assertEqual(c.params(state)['conf'],.04)

if __name__ == '__main__':
    unittest.main()
