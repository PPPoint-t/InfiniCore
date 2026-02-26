import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Applies Group Normalization."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.group_norm(
            input, num_groups, weight=weight, bias=bias, eps=eps
        )

    weight_underlying = weight._underlying if weight is not None else None
    bias_underlying = bias._underlying if bias is not None else None

    # Call C++ Backend via PyBind
    return Tensor(
        _infinicore.group_norm(
            input._underlying, 
            num_groups, 
            weight_underlying, 
            bias_underlying, 
            eps
        )
    )