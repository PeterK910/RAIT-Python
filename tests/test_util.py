import torch
from .util import addimag
import pytest

#addimag test
def test_addimag():
    v1 = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(addimag(v1), torch.tensor([2.+0.j, 2.+0.j, 2.+0.j]))
    v2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert torch.equal(addimag(v2), torch.tensor([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]))
    v3 = torch.tensor([-1.0, 0.0, 1.0, 4.0])
    assert torch.equal(addimag(v3), torch.tensor([ 0.+2.j, -1.-1.j,  2.-2.j,  3.+1.j]))

    #input validation
    with pytest.raises(TypeError, match="v must have real elements."):
        v4 = torch.tensor([1+0j, 2, 3])
        addimag(v4)
    with pytest.raises(ValueError, match="v must be a 1-dimensional torch.Tensor."):
        v5 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        addimag(v5)
