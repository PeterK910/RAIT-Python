import torch
import pytest
import re

from .blaschke import arg_fun
#arg_fun test
def test_arg_fun():
    
    a = torch.tensor([0.5, 0.5, 0.5])
    t = torch.tensor([0.1, 0.2, 0.3])
    expected_result = torch.tensor([0.2980, 0.5848, 0.8514])
    assert torch.allclose(arg_fun(a, t), expected_result)

    a = torch.tensor([0,0,0])
    t = torch.tensor([0,0.5,1,1.5])
    expected_result = torch.tensor([0,0.5,1,1.5])
    assert torch.allclose(arg_fun(a, t), expected_result)

    #input validation
    regex = re.compile(re.escape('"a" must be a (1-dimensional) torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = [1, 2, 3]
        t = torch.tensor([1.0, 2.0, 3.0])
        arg_fun(a, t)

    regex = re.compile(re.escape('"a" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        t = torch.tensor([1.0, 2.0, 3.0])
        arg_fun(a, t)

    regex = re.compile(re.escape('"t" must be a (1-dimensional) torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5])
        t = [1, 2, 3]
        arg_fun(a, t)

    regex = re.compile(re.escape('"t" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5])
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        arg_fun(a, t)

    regex = re.compile(re.escape('Elements of "a" must be inside the unit circle!'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 1])
        t = torch.tensor([1.0, 2.0, 3.0])
        arg_fun(a, t)

    regex = re.compile(re.escape('Elements of "t" must be in [-pi, pi).'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5])
        t = torch.tensor([1.0, 2.0, torch.pi])
        arg_fun(a, t)
    try:
        a = torch.tensor([0.0, 0.0, 0.0])
        t = torch.tensor([-torch.pi, 2.0, 3.0]) # -pi is valid
        arg_fun(a, t)
    except ValueError:
        pytest.fail('arg_fun raised ValueError unexpectedly!')
    

    