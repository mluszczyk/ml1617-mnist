from unittest import TestCase

import numpy

from lib import apply_by_row


class TestApplyByRow(TestCase):
    def test_call(self):
        a = numpy.array([[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]])

        def func(a):
            assert a.shape == (2, 3)
            return a[:, :2]

        b = apply_by_row(func, a)
        self.assertEqual(b.shape, (2, 2, 2))
