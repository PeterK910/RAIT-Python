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
    with pytest.raises(TypeError, match="len must be an integer."):
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mt_system(2.3, poles)

    with pytest.raises(ValueError, match="len must be an integer greater than or equal to 2."):
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mt_system(1, poles)