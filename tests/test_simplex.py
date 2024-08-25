import torch
import pytest
import re 

def test_coords2params():
    from rait.simplex import coords2params
    k=torch.tensor([3,3,1,0,-1,4],dtype=torch.float64)
    expected_result=torch.tensor([0.688247+0.688247j,0.707107+0.000000j,-0.235702+0.942809j],dtype=torch.complex64)
    assert torch.allclose(coords2params(k),expected_result)

    #input validation
    with pytest.raises(TypeError, match="Input k must be a torch.Tensor"):
        k=3
        coords2params(k)
    with pytest.raises(TypeError, match="Input k must be a float tensor"):
        k=torch.tensor([3,3,1,0,-1,4],dtype=torch.int)
        coords2params(k)
    with pytest.raises(ValueError, match="Input k must be a 1D row vector with an even number of elements"):
        k=torch.tensor([3,3,1,0,-1],dtype=torch.float64)
        coords2params(k)
    with pytest.raises(ValueError, match="Input k must be a 1D row vector with an even number of elements"):
        k=torch.tensor([[3,3,1,0,-1,4],[3,3,1,0,-1,4]],dtype=torch.float64)
        coords2params(k)

def test_coords2params_all():
    from rait.simplex import coords2params_all
    k=torch.tensor([[1,1,0,1,-1,0],[1,1,0,1,-1,-1]],dtype=torch.float64)
    expected_result=torch.tensor(
        [[ 0.577350+0.577350j,0.000000+0.707107j,-0.707107+0.000000j],
        [ 0.577350+0.577350j,0.000000+0.707107j,-0.577350+-0.577350j]],dtype=torch.complex64)
    assert torch.allclose(coords2params_all(k),expected_result)

    #input validation
    with pytest.raises(TypeError, match="k must be a torch.Tensor."):
        k=3
        coords2params_all(k)
    with pytest.raises(TypeError, match="k must be a float tensor."):
        k=torch.tensor([[1,1,0,1,-1,0],[1,1,0,1,-1,-1]],dtype=torch.int)
        coords2params_all(k)
    with pytest.raises(ValueError, match="k must be a 2D tensor with an even number of columns"):
        k=torch.tensor([1,1,0,1,-1,0],dtype=torch.float64)
        coords2params_all(k)
    with pytest.raises(ValueError, match="k must be a 2D tensor with an even number of columns"):
        k=torch.tensor([[1,1,0,1,-1],[1,1,0,1,-1]],dtype=torch.float64)
        coords2params_all(k)

def test_multiply_poles():
    from rait.simplex import multiply_poles

    p = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
    m=torch.tensor([1,0,3], dtype=torch.int64)
    expected_result=torch.tensor([0.0000-0.5000j, 0.5000+0.0000j, 0.5000+0.0000j, 0.5000+0.0000j])
    assert torch.allclose(multiply_poles(p,m),expected_result)

    #input validation
    #p is already tested with check_poles(p) in simplex.py
    with pytest.raises(TypeError, match="m must be a torch.Tensor."):
        p = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        m = 3
        multiply_poles(p,m)
    with pytest.raises(TypeError, match="m must be a tensor of dtype int64."):
        p = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        m = torch.tensor([0], dtype=torch.int32)
        multiply_poles(p,m)
    with pytest.raises(ValueError, match="m must be a 1D tensor."):
        p = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        m = torch.tensor(0, dtype=torch.int64)
        multiply_poles(p,m)
    with pytest.raises(ValueError, match="m must contain only non-negative elements."):
        p = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        m = torch.tensor([1,0,-1], dtype=torch.int64)
        multiply_poles(p,m)
    with pytest.raises(ValueError, match="Length of p and m must be equal."):
        p = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        m = torch.tensor([1,0], dtype=torch.int64)
        multiply_poles(p,m)
    with pytest.raises(ValueError, match="Poles in p must be unique."):
        p = torch.tensor([-0.5j,0,-0.5j], dtype=torch.complex64)
        m = torch.tensor([1,0,1], dtype=torch.int64)
        multiply_poles(p,m)
