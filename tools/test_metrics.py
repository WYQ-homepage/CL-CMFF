"""Hand-calculated regression for dropped FP/FN and full OA population."""
import unittest

import numpy as np

from utils.metrics import calculate


class MetricTests(unittest.TestCase):
    def test_cross_excluded_errors(self):
        cm = np.array([[8,1,3],[2,6,4],[5,2,9]])
        before = calculate(cm,[0,1],wrong=True)
        after = calculate(cm,[0,1])
        self.assertAlmostEqual(before['per_class'][0]['iou'],8/11*100)
        self.assertAlmostEqual(after['per_class'][0]['iou'],8/19*100)
        self.assertAlmostEqual(before['per_class'][0]['f1'],16/19*100)
        self.assertAlmostEqual(after['per_class'][0]['f1'],16/27*100)
        self.assertAlmostEqual(before['summary']['oa'],14/17*100)
        self.assertAlmostEqual(after['summary']['oa'],23/40*100)
        self.assertEqual(before['summary']['discarded_pixels'],23)
        self.assertEqual(after['per_class'][0]['fp'],7)
        self.assertEqual(after['per_class'][0]['fn'],4)

    def test_perfect_and_missing_class(self):
        result = calculate(np.diag([10,20,30]),[0,1])
        self.assertEqual(result['summary']['miou'],100)
        self.assertEqual(result['summary']['oa'],100)
        single = calculate(np.diag([10,0,0]),[0,1])
        self.assertEqual(single['summary']['miou'],50)
        self.assertIsNone(single['summary']['kappa'])

    def test_empty_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate(np.zeros((3,3),dtype=int),[0,1])

    def test_bug_cannot_improve_denominators(self):
        rng = np.random.default_rng(42)
        for classes in [6,8]:
            for _ in range(100):
                cm = rng.integers(0,100,size=(classes,classes))
                ids = list(range(classes-1))
                before,after = calculate(cm,ids,wrong=True),calculate(cm,ids)
                self.assertEqual(len(after['per_class']),classes-1)
                for old,new in zip(before['per_class'],after['per_class']):
                    for key in ['iou','f1','precision','recall']:
                        self.assertGreaterEqual(old[key]+1e-10,new[key])


if __name__ == '__main__':
    unittest.main()
