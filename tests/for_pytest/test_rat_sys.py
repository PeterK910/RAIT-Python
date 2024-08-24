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

def test_mlfdc_system():
    from .rat_sys import mlfdc_system
    
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor(
        [[0.800000-4.000000e-01j, 1.707381+5.518449e-01j, 0.743600+3.109000e-01j, 0.800000-3.999996e-01j],
        [1.000000+0.000000e+00j, 1.000000+0.000000e+00j, 1.000000+0.000000e+00j, 1.000000+0.000000e+00j],
        [0.666667-0.000000e+00j, 0.913357-5.177496e-01j, 1.780760+4.942200e-01j, 0.666667+3.017830e-07j]])
    assert torch.allclose(mlfdc_system(mpoles), expected_result)

    #input validation
    #mpoles is already tested with check_poles(mpoles) in rat_sys.py

    with pytest.raises(TypeError, match="eps must be a float."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlfdc_system(mpoles, eps=1)
    with pytest.raises(ValueError, match="eps must be positive."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlfdc_system(mpoles, eps=0.)
    with pytest.raises(ValueError, match="eps must be positive."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlfdc_system(mpoles, eps=-1.)

def test_mlf_coeffs():
    from .rat_sys import mlf_coeffs
    v = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor([-1.060232-0.164866j,0.697041-0.375806j,-0.303476+1.207338j], dtype=torch.complex64)
    co,err = mlf_coeffs(v, mpoles)
    err = torch.tensor(err)
    assert torch.allclose(co, expected_result)
    assert torch.allclose(err, torch.tensor(1.767296))

    #input validation
    with pytest.raises(TypeError, match="v must be a torch.Tensor."):
        v = [2j, 0, -2]
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlf_coeffs(v, mpoles)
    with pytest.raises(ValueError, match="v must be a 1D tensor."):
        v = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlf_coeffs(v, mpoles)
    with pytest.raises(TypeError, match="v must be a complex tensor."):
        v = torch.tensor([2, 0, -2], dtype=torch.float32)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlf_coeffs(v, mpoles)
    #mpoles is already tested with check_poles(mpoles) in rat_sys.py
    