import torch
import pytest

from .blaschke import arg_fun
#arg_fun test
def test_arg_fun():
    
    #input validation
    with pytest.raises(TypeError, match='"a" must be a (1-dimensional) torch.Tensor.'):
        a = [1, 2, 3]
        t = torch.tensor([1.0, 2.0, 3.0])
        arg_fun(a, t)
    with pytest.raises(ValueError, match='"a" must be a 1-dimensional torch.Tensor.'):
        a = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        t = torch.tensor([1.0, 2.0, 3.0])
        arg_fun(a, t)
    with pytest.raises(TypeError, match='"t" must be a (1-dimensional) torch.Tensor.'):
        a = torch.tensor([0.5, 0.5, 0.5])
        t = [1, 2, 3]
        arg_fun(a, t)
    with pytest.raises(ValueError, match='"t" must be a 1-dimensional torch.Tensor.'):
        a = torch.tensor([0.5, 0.5, 0.5])
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        arg_fun(a, t)
    with pytest.raises(ValueError, match='Elements of "a" must be inside the unit circle!'):
        a = torch.tensor([0.5, 0.5, 1])
        t = torch.tensor([1.0, 2.0, 3.0])
        arg_fun(a, t)
    with pytest.raises(ValueError, match='Elements of "t" must be in [-pi, pi).'):
        a = torch.tensor([0.5, 0.5, 0.5])
        t = torch.tensor([1.0, 2.0, torch.pi])
        arg_fun(a, t)
    try:
        a = torch.tensor([1.0, 0.0, 0.0])
        t = torch.tensor([-torch.pi, 2.0, 3.0])
        arg_fun(a, t)
    except ValueError:
        pytest.fail('arg_fun raised ValueError unexpectedly!')