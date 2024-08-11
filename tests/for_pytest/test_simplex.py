import torch
import pytest
import re 

def test_coords2params():
    from .simplex import coords2params
    k=torch.tensor([3,3,1,0,-1,4],dtype=torch.float64)
    expected_result=torch.tensor([0.688247+0.688247j,0.707107+0.000000j,-0.235702+0.942809j],dtype=torch.complex64)
    assert torch.allclose(coords2params(k),expected_result)

    #input validation
    with pytest.raises(TypeError, match="Input k must be a torch.Tensor"):
        k=3
        coords2params(k)
    with pytest.raises(TypeError, match="Input k must be a tensor of dtype float64"):
        k=torch.tensor([3,3,1,0,-1,4],dtype=torch.float32)
        coords2params(k)
    with pytest.raises(ValueError, match="Input k must be a 1D row vector with an even number of elements"):
        k=torch.tensor([3,3,1,0,-1],dtype=torch.float64)
        coords2params(k)
    with pytest.raises(ValueError, match="Input k must be a 1D row vector with an even number of elements"):
        k=torch.tensor([[3,3,1,0,-1,4],[3,3,1,0,-1,4]],dtype=torch.float64)
        coords2params(k)

def test_coords2params_all():
    from .simplex import coords2params_all
    k=torch.tensor([[1,1,0,1,-1,0],[1,1,0,1,-1,-1]],dtype=torch.float64)
    expected_result=torch.tensor(
        [[ 0.577350+0.577350j,0.000000+0.707107j,-0.707107+0.000000j],
        [ 0.577350+0.577350j,0.000000+0.707107j,-0.577350+-0.577350j]],dtype=torch.complex64)
    assert torch.allclose(coords2params_all(k),expected_result)

    #input validation
    with pytest.raises(TypeError, match="k must be a torch.Tensor."):
        k=3
        coords2params_all(k)
    with pytest.raises(TypeError, match="k must be a tensor of dtype float64"):
        k=torch.tensor([[1,1,0,1,-1,0],[1,1,0,1,-1,-1]],dtype=torch.float32)
        coords2params_all(k)
    with pytest.raises(ValueError, match="k must be a 2D tensor with an even number of columns"):
        k=torch.tensor([1,1,0,1,-1,0],dtype=torch.float64)
        coords2params_all(k)
    with pytest.raises(ValueError, match="k must be a 2D tensor with an even number of columns"):
        k=torch.tensor([[1,1,0,1,-1],[1,1,0,1,-1]],dtype=torch.float64)
        coords2params_all(k)