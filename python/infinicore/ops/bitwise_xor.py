import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def bitwise_xor(input: Tensor, other: Tensor, out: Tensor | None = None) -> Tensor:
    """Computes the bitwise XOR of input and other."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.bitwise_xor(input, other, out=out)
    
    if out is not None:
        _infinicore.bitwise_xor_out(input._underlying, other._underlying, out._underlying)
        return out
    else:
        return Tensor(_infinicore.bitwise_xor(input._underlying, other._underlying))