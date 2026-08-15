import sys
import types
import unittest
from unittest.mock import patch

from molmo_quant.compatibility import patch_bnb_linear8bitlt_high_rank


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def dim(self):
        return len(self.shape)

    def reshape(self, *shape):
        resolved = list(shape)
        if -1 in resolved:
            known = 1
            for value in resolved:
                if value != -1:
                    known *= value
            total = 1
            for value in self.shape:
                total *= value
            resolved[resolved.index(-1)] = total // known
        return FakeTensor(resolved)


class FakeLinear8bitLt:
    def forward(self, inputs):
        return inputs


class CompatibilityTests(unittest.TestCase):
    def test_high_rank_patch_preserves_leading_shape_and_is_idempotent(self):
        fake_bnb = types.SimpleNamespace(
            nn=types.SimpleNamespace(Linear8bitLt=FakeLinear8bitLt)
        )
        with patch.dict(sys.modules, {"bitsandbytes": fake_bnb}):
            self.assertTrue(patch_bnb_linear8bitlt_high_rank())
            self.assertFalse(patch_bnb_linear8bitlt_high_rank())
            output = FakeLinear8bitLt().forward(FakeTensor((2, 3, 4, 5)))
        self.assertEqual(output.shape, (2, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()
