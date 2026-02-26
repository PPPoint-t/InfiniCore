import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def rrelu(input: Tensor, lower: float = 0.125, upper: float = 0.3333333333333333, training: bool = False, inplace: bool = False, generator=None) -> Tensor:
    r"""Applies the randomized leaky rectified linear unit function."""
    
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.rrelu(input, lower, upper, training, inplace, generator)
    
    return Tensor(_infinicore.rrelu(input._underlying, lower, upper, training, inplace))