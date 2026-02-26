import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def isclose(
    input: Tensor, 
    other: Tensor, 
    rtol: float = 1e-05, 
    atol: float = 1e-08, 
    equal_nan: bool = False
) -> Tensor:
    r"""Returns a boolean tensor where two tensors are element-wise equal within a tolerance."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.isclose(input, other, rtol, atol, equal_nan)
    
    return Tensor(_infinicore.isclose(input._underlying, other._underlying, rtol, atol, equal_nan))