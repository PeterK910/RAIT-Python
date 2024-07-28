import torch
import pytest
import re 

def test_biort_system():
    from .biort_sys import biort_system

    length = 4
    mpoles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
    expected_result = torch.tensor(
        [[-1.6500-1.0500j,  4.3500+0.4500j, -0.1500-1.9500j, -0.5500+1.3500j],
        [ 3.2000+2.4000j, -3.2000-2.4000j, -3.2000+2.4000j,  3.2000-2.4000j],
        [-0.5500-1.3500j, -0.1500+1.9500j,  4.3500-0.4500j, -1.6500+1.0500j]])
    assert torch.allclose(biort_system(length, mpoles), expected_result)

    #input validation
    with pytest.raises(TypeError, match="Length must be an integer."):
        biort_system(2.3, mpoles)

    with pytest.raises(ValueError, match="Length must be at least 2."):
        biort_system(1, mpoles)

    with pytest.raises(ValueError, match="Length must be at least 2."):
        biort_system(-1, mpoles)
