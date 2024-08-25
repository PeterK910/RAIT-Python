import torch
import pytest
import re


#arg_fun test
def test_arg_fun():
    from rait.blaschke import arg_fun

    a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
    t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([0.298025, 0.584755, 0.851365], dtype=torch.float64)
    assert torch.allclose(arg_fun(a, t), expected_result)

    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result=torch.tensor([0.770113, 0.916895, 1.055801], dtype=torch.float64)
    assert torch.allclose(arg_fun(a, t), expected_result)

    #test if t is only used in trigonometric functions
    a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
    t2=torch.tensor([0.1 + 2*torch.pi, 0.2, 0.3], dtype=torch.float64)
    expected_result2 = torch.tensor([0.298025, 0.584755, 0.851365], dtype=torch.float64)
    assert torch.allclose(arg_fun(a, t2), expected_result2)
    
    a = torch.tensor([0,0,0], dtype=torch.complex64)
    t = torch.tensor([0,0.5,1,1.5], dtype=torch.float64)
    expected_result = torch.tensor([0,0.5,1,1.5], dtype=torch.float64)
    assert torch.allclose(arg_fun(a, t), expected_result)

    #edge case where t has a -pi value
    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    t = torch.tensor([-torch.pi, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-3.141593,  0.916895,  1.055801], dtype=torch.float64)
    assert torch.allclose(arg_fun(a, t), expected_result)

    #input validation
    #a is already tested with check_poles(a) in blaschke.py

    #t
    regex = re.compile(re.escape('"t" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = [1, 2, 3]
        arg_fun(a, t)

    regex = re.compile(re.escape('"t" must be a float torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3], dtype=torch.int)
        arg_fun(a, t)

    regex = re.compile(re.escape('"t" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        arg_fun(a, t)
   
def test_argdr_fun():
    from rait.blaschke import argdr_fun

    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([2.310337, 2.750686, 3.167403], dtype=torch.float64)
    assert torch.allclose(argdr_fun(a, t), expected_result)

    #test if t is only used in trigonometric functions
    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    t2=torch.tensor([0.1 + 2*torch.pi, 0.2, 0.3], dtype=torch.float64)
    expected_result2 = torch.tensor([21.15989, 2.750686, 3.167403], dtype=torch.float64)
    assert torch.allclose(argdr_fun(a, t2), expected_result2)

    #input validation
    #a is already tested with check_poles(a) in blaschke.py

    #t
    regex = re.compile(re.escape('"t" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = [1, 2, 3]
        argdr_fun(a, t)

    regex = re.compile(re.escape('"t" must be a float torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3], dtype=torch.int)
        argdr_fun(a, t)

    regex = re.compile(re.escape('"t" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        argdr_fun(a, t)

def test_argdr_inv():
    from rait.blaschke import argdr_inv
    #when first parameter is 1 number
    a = torch.tensor([-0.5j], dtype=torch.complex64)
    b = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-1.312326, -1.273881, -1.233906], dtype=torch.float64)
    assert torch.allclose(argdr_inv(a, b), expected_result)

    #edge case where b has a -pi value
    a = torch.tensor([-0.5j], dtype=torch.complex64)
    b = torch.tensor([-torch.pi, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-3.141593, -1.273881, -1.233906], dtype=torch.float64)
    assert torch.allclose(argdr_inv(a, b), expected_result)

    #when first parameter is more than 1 number
    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    b = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-0.392699, -0.312549, -0.235124], dtype=torch.float64)
    assert torch.allclose(argdr_inv(a, b), expected_result)

    #edge case where b has a -pi value
    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    b = torch.tensor([-torch.pi, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-3.141593, -0.312549, -0.235071], dtype=torch.float64)
    assert torch.allclose(argdr_inv(a, b), expected_result)
    #input validation
    #a is already tested with check_poles(a) in blaschke.py

    #b
    regex = re.compile(re.escape('"b" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = [1, 2, 3]
        argdr_inv(a, b)

    regex = re.compile(re.escape('"b" must be a float torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([1, 2, 3], dtype=torch.int)
        argdr_inv(a, b)

    regex = re.compile(re.escape('"b" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        argdr_inv(a, b)

    #epsi
    regex = re.compile(re.escape('"epsi" must be a float.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_inv(a, b, epsi=1)
    
    regex = re.compile(re.escape('"epsi" must be a positive float.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        argdr_inv(a, b, epsi=-1.)

def test_arg_der():
    from rait.blaschke import arg_der
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

    regex = re.compile(re.escape('"t" must be a float torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([0.5, 0.5, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3], dtype=torch.int)
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

def test_arg_inv():
    from rait.blaschke import arg_inv
    #when first parameter is 1 number
    a = torch.tensor([-0.5j], dtype=torch.complex64)
    b = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-1.312326, -1.273881, -1.233906], dtype=torch.float64)
    assert torch.allclose(arg_inv(a, b), expected_result)

    #edge case where b has a -pi value
    a = torch.tensor([-0.5j], dtype=torch.complex64)
    b = torch.tensor([-torch.pi, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-3.141593, -1.273881, -1.233906], dtype=torch.float64)
    assert torch.allclose(arg_inv(a, b), expected_result)

    #when first parameter is more than 1 number
    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    b = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-0.346104, -0.276500, -0.208710], dtype=torch.float64)
    assert torch.allclose(arg_inv(a, b), expected_result)

    #edge case where b has a -pi value
    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    b = torch.tensor([-torch.pi, 0.2, 0.3], dtype=torch.float64)
    expected_result = torch.tensor([-3.141593, -0.276500, -0.208697], dtype=torch.float64)
    assert torch.allclose(arg_inv(a, b), expected_result)
    #input validation
    #a is already tested with check_poles(a) in blaschke.py

    #b
    regex = re.compile(re.escape('"b" must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = [1, 2, 3]
        arg_inv(a, b)

    regex = re.compile(re.escape('"b" must be a float torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([1, 2, 3], dtype=torch.int)
        arg_inv(a, b)

    regex = re.compile(re.escape('"b" must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        arg_inv(a, b)

    #epsi
    regex = re.compile(re.escape('"epsi" must be a float.'))
    with pytest.raises(TypeError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_inv(a, b, epsi=1)
    
    regex = re.compile(re.escape('"epsi" must be a positive float.'))
    with pytest.raises(ValueError, match=regex):
        a = torch.tensor([-0.5j], dtype=torch.complex64)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        arg_inv(a, b, epsi=-1.)
def test_blaschkes():
    from rait.blaschke import blaschkes
    len=3
    poles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
    expected_result = torch.tensor(
        [-0.6000+0.8000j, -0.976627+0.214941j, -0.177219-0.984171j, -0.6000+0.8000j], dtype=torch.complex64)
    assert torch.allclose(blaschkes(len, poles), expected_result)

    #input validation
    regex = re.compile(re.escape('"len" must be an integer.'))
    with pytest.raises(TypeError, match=regex):
        poles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        blaschkes(2.3, poles)
    regex = re.compile(re.escape('"len" must be a positive integer.'))
    with pytest.raises(ValueError, match=regex):
        poles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        blaschkes(0, poles)