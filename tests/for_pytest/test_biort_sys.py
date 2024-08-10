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
    #mpoles is already tested with check_poles(mpoles) in biort_sys.py
    with pytest.raises(TypeError, match="Length must be an integer."):
        biort_system(2.3, mpoles)

    with pytest.raises(ValueError, match="Length must be at least 2."):
        biort_system(1, mpoles)

    with pytest.raises(ValueError, match="Length must be at least 2."):
        biort_system(-1, mpoles)
    

def test_biortdc_system():
    from .biort_sys import biortdc_system
    mpoles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
    expected_result = torch.tensor(
        [[-1.650000-1.050000j,  2.082207+3.325435j,  1.467794-0.975437j, -1.650001-1.049997j],
        [ 3.200000+2.400000j,  1.157638-3.828822j, -3.757642-1.371177j, 3.200003+2.399996j],
        [-0.550000-1.350000j, -2.239845+0.503386j,  3.289847+2.346614j, -0.550002-1.349999j]])
    assert torch.allclose(biortdc_system(mpoles), expected_result)

    #input validation
    #mpoles is already tested with check_poles(mpoles) in biort_sys.py
    with pytest.raises(TypeError, match="eps must be a float."):
        mpoles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        biortdc_system(mpoles, eps=1)
    with pytest.raises(ValueError, match="eps must be positive."):
        mpoles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        biortdc_system(mpoles, eps=0.)