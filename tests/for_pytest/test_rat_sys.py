import torch
import pytest
import re 

def test_mlf_system():
    from .rat_sys import mlf_system
    len=5
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor(
        [[0.800000-4.000000e-01j, 1.754418-5.168485e-01j, 1.066282+6.108419e-01j, 0.704050+2.201065e-01j, 0.670373-7.019743e-02j],
        [1.000000+0.000000e+00j, 1.000000+0.000000e+00j, 1.000000+0.000000e+00j, 1.000000+0.000000e+00j, 1.000000+0.000000e+00j],
        [0.666667+1.942728e-08j, 0.740536-3.050180e-01j, 1.350373-6.664488e-01j, 1.350373+6.664488e-01j, 0.740536+3.050180e-01j]])
    assert torch.allclose(mlf_system(len, mpoles), expected_result)

    #input validation
    with pytest.raises(TypeError, match="Length must be an integer."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlf_system(2.3, mpoles)
    with pytest.raises(ValueError, match="Length must be greater than or equal to 2."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlf_system(1, mpoles)
    
    #mpoles is already tested with check_poles(mpoles) in rat_sys.py

def test_lf_system():
    from .rat_sys import lf_system
    len=5
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor(
        [[0.693497-3.467486e-01j, 1.520855-4.480413e-01j, 0.924329+5.295214e-01j, 0.610321+1.908041e-01j, 0.581127-6.085215e-02j],
        [1.000000+0.000000e+00j, 1.000000+0.000000e+00j, 1.000000+0.000000e+00j, 1.000000+0.000000e+00j, 1.000000+0.000000e+00j],
        [0.595683+0.000000e+00j, 0.661688-2.725412e-01j, 1.206592-5.954888e-01j, 1.206592+5.954888e-01j, 0.661688+2.725412e-01j]])
    assert torch.allclose(lf_system(len, poles), expected_result)

    #input validation
    with pytest.raises(TypeError, match="Length must be an integer."):
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        lf_system(2.3, poles)
    with pytest.raises(ValueError, match="Length must be greater than or equal to 2."):
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        lf_system(1, poles)
    
    #mpoles is already tested with check_poles(mpoles) in rat_sys.py