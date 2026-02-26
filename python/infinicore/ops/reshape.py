import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def reshape(input: Tensor, shape: tuple | list) -> Tensor:
    r"""
    Returns a tensor with the same data and number of elements as input, 
    but with the specified shape.
    """

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.reshape(input, shape)

    return Tensor(_infinicore.reshape(input._underlying, shape))