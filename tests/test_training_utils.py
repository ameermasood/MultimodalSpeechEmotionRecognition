import os
import random
import tempfile
import unittest

import numpy as np
import torch

from mer.training.utils import safe_makedirs, set_seed, to_abs


class TrainingUtilsTests(unittest.TestCase):
    def test_to_abs_expands_user_and_relative_paths(self):
        path = to_abs(".")

        self.assertTrue(os.path.isabs(path))

    def test_safe_makedirs_creates_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "nested", "output")

            safe_makedirs(target)

            self.assertTrue(os.path.isdir(target))

    def test_set_seed_repeats_random_sequences(self):
        set_seed(123)
        first = (random.random(), np.random.rand(), torch.rand(1).item())

        set_seed(123)
        second = (random.random(), np.random.rand(), torch.rand(1).item())

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
