import torch
import pytest
import re


#arg_fun test
def test_arg_fun():
    from .blaschke import arg_fun

    a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
    t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([0.298025, 0.584755, 0.851365], dtype=torch.float64)
    assert torch.allclose(arg_fun(a, t), expected_result)

    a = torch.tensor([0,0,0], dtype=torch.complex64)
    t = torch.tensor([0,0.5,1,1.5], dtype=torch.float64)
    expected_result = torch.tensor([0,0.5,1,1.5], dtype=torch.float64)
    assert torch.allclose(arg_fun(a, t), expected_result)
    

    #input validation
    #a
    regex = re.compile(re.escape('"a" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = [1, 2, 3]
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_fun(a, t)

    regex = re.compile(re.escape('"a" must be a complex torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([1.0, 2.0, 3.0])
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_fun(a, t)

    regex = re.compile(re.escape('"a" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_fun(a, t)

    regex = re.compile(re.escape('Elements of "a" must be inside the unit circle!'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 1], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_fun(a, t)

    #t
    regex = re.compile(re.escape('"t" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = [1, 2, 3]
        arg_fun(a, t)

    regex = re.compile(re.escape('"t" must be a torch.Tensor with float64 dtype.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        arg_fun(a, t)

    regex = re.compile(re.escape('"t" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        arg_fun(a, t)

    regex = re.compile(re.escape('Elements of "t" must be in [-pi, pi).'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, torch.pi], dtype=torch.float64)
        arg_fun(a, t)
    try:
        a = torch.tensor([0, 0, 0], dtype=torch.complex64)
        t = torch.tensor([-torch.pi, 2.0, 3.0], dtype=torch.float64) # -pi is valid
        arg_fun(a, t)
    except ValueError:
        pytest.fail('arg_fun raised ValueError unexpectedly!')
   
def test_argdr_fun():
    from .blaschke import argdr_fun

    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([2.310337, 2.750686, 3.167403], dtype=torch.float64)
    assert torch.allclose(argdr_fun(a, t), expected_result)

    #input validation
    #a
    regex = re.compile(re.escape('"a" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = [1, 2, 3]
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_fun(a, t)

    regex = re.compile(re.escape('"a" must be a complex torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([1.0, 2.0, 3.0])
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_fun(a, t)

    regex = re.compile(re.escape('"a" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_fun(a, t)

    regex = re.compile(re.escape('Elements of "a" must be inside the unit circle!'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 1], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_fun(a, t)

    #t
    regex = re.compile(re.escape('"t" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = [1, 2, 3]
        argdr_fun(a, t)

    regex = re.compile(re.escape('"t" must be a torch.Tensor with float64 dtype.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        argdr_fun(a, t)

    regex = re.compile(re.escape('"t" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        argdr_fun(a, t)

    regex = re.compile(re.escape('Elements of "t" must be in [-pi, pi).'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, torch.pi], dtype=torch.float64)
        argdr_fun(a, t)
    try:
        a = torch.tensor([0, 0, 0], dtype=torch.complex64)
        t = torch.tensor([-torch.pi, 2.0, 3.0], dtype=torch.float64) # -pi is valid
        argdr_fun(a, t)
    except ValueError:
        pytest.fail('argdr_fun raised ValueError unexpectedly!')

def test_argdr_inv():
    from .blaschke import argdr_inv
    #when first parameter is 1 number
    a = torch.tensor([-0.5j], dtype=torch.complex64)
    b = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-1.312326, -1.273881, -1.233906], dtype=torch.float64)
    assert torch.allclose(argdr_inv(a, b), expected_result)

    #when first parameter is more than 1 number
    #TODO: consult with the author about the expected result, see 
    #https://github.com/Nguyen-Thac-Bach/RAIT-Python/issues/10

    #input validation
    #a
    regex = re.compile(re.escape('"a" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = [1, 2, 3]
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_inv(a, b)

    regex = re.compile(re.escape('"a" must be a complex torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_inv(a, b)
    
    regex = re.compile(re.escape('"a" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.complex64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_inv(a, b)

    regex = re.compile(re.escape('Elements of "a" must be inside the unit circle!'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 1], dtype=torch.complex64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_inv(a, b)

    #b
    regex = re.compile(re.escape('"b" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = [1, 2, 3]
        argdr_inv(a, b)

    regex = re.compile(re.escape('"b" must be a torch.Tensor with float64 dtype.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        argdr_inv(a, b)

    regex = re.compile(re.escape('"b" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        argdr_inv(a, b)

    regex = re.compile(re.escape('Elements of "b" must be in [-pi, pi).'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([1.0, 2.0, torch.pi], dtype=torch.float64)
        argdr_inv(a, b)

    try:
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([-torch.pi, 2.0, 3.0], dtype=torch.float64) # -pi is valid
        argdr_inv(a, b)
    except ValueError:
        pytest.fail('argdr_inv raised ValueError unexpectedly!')

def test_arg_der():
    from .blaschke import arg_der
    a = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
    t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([1.498950, 1.432060, 1.343517], dtype=torch.float64)
    assert torch.allclose(arg_der(a, t), expected_result)

    #input validation
    #a
    regex = re.compile(re.escape('"a" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = [1, 2, 3]
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_der(a, t)

    regex = re.compile(re.escape('"a" must be a complex torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([1.0, 2.0, 3.0])
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_der(a, t)

    regex = re.compile(re.escape('"a" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_der(a, t)

    regex = re.compile(re.escape('Elements of "a" must be inside the unit circle!'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 1], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_der(a, t)

    #t
    regex = re.compile(re.escape('"t" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = [1, 2, 3]
        arg_der(a, t)

    regex = re.compile(re.escape('"t" must be a torch.Tensor with float64 dtype.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        arg_der(a, t)

    regex = re.compile(re.escape('"t" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        arg_der(a, t)

    regex = re.compile(re.escape('Elements of "t" must be in [-pi, pi).'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([1.0, 2.0, torch.pi], dtype=torch.float64)
        arg_der(a, t)
    try:
        a = torch.tensor([0, 0, 0], dtype=torch.complex64)
        t = torch.tensor([-torch.pi, 2.0, 3.0], dtype=torch.float64) # -pi is valid
        arg_der(a, t)
    except ValueError:
        pytest.fail('argdr_fun raised ValueError unexpectedly!')