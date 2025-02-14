import torch
import pytest
import re 

def test_mlf_system():
    from rait.rat_sys import mlf_system
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
    from rait.rat_sys import lf_system
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
    from rait.rat_sys import mlfdc_system
    
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
    from rait.rat_sys import mlf_coeffs
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
    
def test_mlfdc_coeffs():
    from rait.rat_sys import mlfdc_coeffs
    signal = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    co,err = mlfdc_coeffs(signal, mpoles)
    err = torch.tensor(err)
    expected_co = torch.tensor([1.162314+-2.456568j,-1.355743+5.941933j,-0.887221+-2.267632j], dtype=torch.complex64)
    expected_err = torch.tensor(3.613619)
    assert torch.allclose(co, expected_co)
    assert torch.allclose(err, expected_err)

    #input validation
    with pytest.raises(TypeError, match="signal must be a torch.Tensor."):
        signal = [2j, 0, -2]
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlfdc_coeffs(signal, mpoles)
    with pytest.raises(ValueError, match="signal must be a 1-dimensional torch.Tensor."):
        signal = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlfdc_coeffs(signal, mpoles)
    with pytest.raises(TypeError, match="signal must be a complex tensor."):
        signal = torch.tensor([2, 0, -2], dtype=torch.float32)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlfdc_coeffs(signal, mpoles)
    #mpoles is already tested with check_poles(mpoles) in rat_sys.py
    #eps
    with pytest.raises(TypeError, match="eps must be a float."):
        signal = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlfdc_coeffs(signal, mpoles, eps=1)
    with pytest.raises(ValueError, match="eps must be a positive float."):
        signal = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mlfdc_coeffs(signal, mpoles, eps=0.)

def test_mlf_generate():
    from rait.rat_sys import mlf_generate

    length=5
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    result = mlf_generate(length, poles, coeffs)
    expected_result=torch.tensor([-0.533333+1.600000j,-0.447375+4.118871j,-3.922430+3.465461j,-3.140959+0.075202j,-1.340678+0.730709j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #input validation
    #length
    with pytest.raises(TypeError, match="length must be an integer."):
        length=2.3
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mlf_generate(length, poles, coeffs)
    with pytest.raises(ValueError, match="length must be greater than or equal to 2."):
        length=1
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mlf_generate(length, poles, coeffs)
    #poles is already tested with check_poles(poles) in rat_sys.py
    #coeffs
    with pytest.raises(TypeError, match="coeffs must be a torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = [2j, 0, -2]
        mlf_generate(length, poles, coeffs)
    with pytest.raises(ValueError, match="coeffs must be a 1-dimensional torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        mlf_generate(length, poles, coeffs)
    with pytest.raises(TypeError, match="coeffs must be a complex tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2, 0, -2], dtype=torch.float32)
        mlf_generate(length, poles, coeffs)
    #coeffs and poles do not have the same length
    with pytest.raises(ValueError, match="poles and coeffs must have the same number of elements."):
        length=5
        poles = torch.tensor([-0.5j, 0], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mlf_generate(length, poles, coeffs)

def test_lf_generate():
    from rait.rat_sys import lf_generate

    length=5
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    result = lf_generate(length, poles, coeffs)
    expected_result = torch.tensor([-0.497870+1.386994j,-0.427293+3.586792j,-3.472227+3.039636j,-2.794792+0.029664j,-1.201671+0.617172j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #input validation
    #length
    with pytest.raises(TypeError, match="length must be an integer."):
        length=2.3
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        lf_generate(length, poles, coeffs)
    with pytest.raises(ValueError, match="length must be greater than or equal to 2."):
        length=1
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        lf_generate(length, poles, coeffs)
    #poles is already tested with check_poles(poles) in rat_sys.py
    #coeffs
    with pytest.raises(TypeError, match="coeffs must be a torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = [2j, 0, -2]
        lf_generate(length, poles, coeffs)
    with pytest.raises(ValueError, match="coeffs must be a 1-dimensional torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        lf_generate(length, poles, coeffs)
    with pytest.raises(TypeError, match="coeffs must be a complex tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2, 0, -2], dtype=torch.float32)
        lf_generate(length, poles, coeffs)
    #coeffs and poles do not have the same length
    with pytest.raises(ValueError, match="poles and coeffs must have the same number of elements."):
        length=5
        poles = torch.tensor([-0.5j, 0], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        lf_generate(length, poles, coeffs)

def test_mlfdc_generate():
    from rait.rat_sys import mlfdc_generate
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    result = mlfdc_generate(mpoles, coeffs)
    expected_result = torch.tensor([-0.533333+1.600000j,-2.930405+4.450261j,-4.183321+0.498760j,-0.533334+1.599999j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #input validation
    #mpoles is already tested with check_poles(mpoles) in rat_sys.py
    #coeffs
    with pytest.raises(TypeError, match="coeffs must be a torch.Tensor."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = [2j, 0, -2]
        mlfdc_generate(mpoles, coeffs)
    with pytest.raises(ValueError, match="coeffs must be a 1-dimensional torch.Tensor."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        mlfdc_generate(mpoles, coeffs)
    with pytest.raises(TypeError, match="coeffs must be a complex tensor."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2, 0, -2], dtype=torch.float32)
        mlfdc_generate(mpoles, coeffs)
    #coeffs and poles do not have the same length
    with pytest.raises(ValueError, match="mpoles and coeffs must have the same number of elements."):
        length=5
        mpoles = torch.tensor([-0.5j, 0], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mlfdc_generate(mpoles, coeffs)
    #eps
    with pytest.raises(TypeError, match="eps must be a float."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mlfdc_generate(mpoles, coeffs, eps=1)
    with pytest.raises(ValueError, match="eps must be a positive float."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mlfdc_generate(mpoles, coeffs, eps=0.)