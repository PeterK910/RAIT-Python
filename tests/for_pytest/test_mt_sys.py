import torch
import pytest
import re 

def test_mt_system():
    from .mt_sys import mt_system
    len=3
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor(
        [[ 0.692820-0.346410j,  1.278796+0.563856j,  0.586489+0.102317j],
        [-0.6000+0.8000j,  0.976627-0.214941j,  0.177219+0.984171j],
        [ 0.346410-0.461880j, -0.214941-0.976627j, -0.984171+0.177219j]], dtype=torch.complex64)
    assert torch.allclose(mt_system(len, poles), expected_result)

    #input validation
    #poles is already tested with check_poles(mpoles) in mt_sys.py
    with pytest.raises(TypeError, match="len must be an integer."):
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mt_system(2.3, poles)

    with pytest.raises(ValueError, match="len must be an integer greater than or equal to 2."):
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mt_system(1, poles)

def test_mtdc_system():
    from .mt_sys import mtdc_system
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor(
        [[ 0.692820-0.346410j,  1.478635+0.477912j,  0.643976+0.269247j, 0.692820-0.346410j],
        [-0.600000+0.800000j,  0.827767-0.561071j,  0.466350+0.884600j, -0.599999+0.800000j],
        [ 0.346410-0.461880j, -0.627374-0.658116j, -0.126577+1.595462j, 0.346409-0.461881j]])
    assert torch.allclose(mtdc_system(mpoles), expected_result)

    #input validation
    #mpoles is already tested with check_poles(mpoles) in mt_sys.py
    with pytest.raises(TypeError, match="eps must be a float"):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdc_system(mpoles, eps=1)
    with pytest.raises(ValueError, match="eps must be positive"):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdc_system(mpoles, eps=-1.)
    with pytest.raises(ValueError, match="eps must be positive"):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdc_system(mpoles, eps=0.)