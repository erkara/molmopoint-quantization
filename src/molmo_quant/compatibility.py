"""Narrow runtime compatibility fixes discovered during prototyping."""

from __future__ import annotations


def patch_bnb_linear8bitlt_high_rank() -> bool:
    """Allow bitsandbytes ``Linear8bitLt`` to accept tensors with rank > 3.

    ``torch.nn.Linear`` applies the matrix multiplication to the last axis.
    Flattening the leading axes before the bitsandbytes call and restoring them
    afterward preserves those semantics. The patch is process-local and
    idempotent.

    Returns ``True`` when this call installed the patch and ``False`` when it
    was already present.
    """

    import bitsandbytes as bnb

    cls = bnb.nn.Linear8bitLt
    if getattr(cls, "_molmo_high_rank_patch", False):
        return False

    original_forward = cls.forward

    def forward(self, inputs):
        if inputs.dim() <= 3:
            return original_forward(self, inputs)
        input_shape = inputs.shape
        outputs = original_forward(self, inputs.reshape(-1, input_shape[-1]))
        return outputs.reshape(*input_shape[:-1], outputs.shape[-1])

    cls.forward = forward
    cls._molmo_high_rank_patch = True
    return True

