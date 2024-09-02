import torch
import pytest
import re 

def test_biort_system():
    from rait.biort_sys import biort_system

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
    from rait.biort_sys import biortdc_system
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

def test_biort_coeffs():
    from rait.biort_sys import biort_coeffs
    v = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    poles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
    expected_co = torch.tensor([-0.718146+0.612097j,-0.666667+0.666667j,-0.666667+0.829345j], dtype=torch.complex64)
    expected_err = torch.tensor(1.767295)
    co,err = biort_coeffs(v, poles)
    err = torch.tensor(err)
    assert torch.allclose(co, expected_co)
    assert torch.allclose(err, expected_err)

    #input validation
    with pytest.raises(TypeError, match="v must be a torch.Tensor."):
        v = [2j, 0, -2]
        poles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        biort_coeffs(v, poles)
    with pytest.raises(ValueError, match="v must be a 1-dimensional torch.Tensor."):
        v = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        poles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        biort_coeffs(v, poles)
    with pytest.raises(TypeError, match="v must be a complex tensor."):
        v = torch.tensor([2, 0, -2], dtype=torch.float32)
        poles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
        biort_coeffs(v, poles)
    #poles is already tested with check_poles(poles) in biort_sys.py

def test_biortdc_coeffs():
    from rait.biort_sys import biortdc_coeffs

    v = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    poles = torch.tensor([-0.5j,0,0.5], dtype=torch.complex64)
    co,err = biortdc_coeffs(v, poles)
    err = torch.tensor(err)
    expected_co = torch.tensor([-0.346997+1.154818j,-0.046167+1.217733j,-0.177091+0.879845j], dtype=torch.complex64)
    expected_err = torch.tensor(3.080572)
    assert torch.allclose(co, expected_co)
    assert torch.allclose(err, expected_err)

    #input validation
    with pytest.raises(TypeError, match="v must be a torch.Tensor."):
        v = [2j, 0, -2]
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        biortdc_coeffs(v, mpoles)
    with pytest.raises(ValueError, match="v must be a 1-dimensional torch.Tensor."):
        v = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        biortdc_coeffs(v, mpoles)
    with pytest.raises(TypeError, match="v must be a complex tensor."):
        v = torch.tensor([2, 0, -2], dtype=torch.float32)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        biortdc_coeffs(v, mpoles)
    #mpoles is already tested with check_poles(mpoles) in rat_sys.py
    #eps
    with pytest.raises(TypeError, match="eps must be a float."):
        v = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        biortdc_coeffs(v, mpoles, eps=1)
    with pytest.raises(ValueError, match="eps must be a positive float."):
        v = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        biortdc_coeffs(v, mpoles, eps=0.)

def test_biort_generate():
    from rait.biort_sys import biort_generate
    length=5
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    result = biort_generate(length, poles, coeffs)
    expected_result = torch.tensor([3.200000+-0.600000j,1.900022+4.552000j,-2.254179+2.161684j,-2.972920+-3.315897j,1.291600+-2.805769j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #input validation
    #length
    with pytest.raises(TypeError, match="length must be an integer."):
        length=2.3
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        biort_generate(length, poles, coeffs)
    with pytest.raises(ValueError, match="length must be greater than or equal to 2."):
        length=1
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        biort_generate(length, poles, coeffs)
    #poles is already tested with check_poles(poles) in rat_sys.py
    #coeffs
    with pytest.raises(TypeError, match="coeffs must be a torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = [2j, 0, -2]
        biort_generate(length, poles, coeffs)
    with pytest.raises(ValueError, match="coeffs must be a 1-dimensional torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        biort_generate(length, poles, coeffs)
    with pytest.raises(TypeError, match="coeffs must be a complex tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2, 0, -2], dtype=torch.float32)
        biort_generate(length, poles, coeffs)
    #coeffs and poles do not have the same length
    with pytest.raises(ValueError, match="poles and coeffs must have the same number of elements."):
        length=5
        poles = torch.tensor([-0.5j, 0], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        biort_generate(length, poles, coeffs)
        
def test_biortdc_generate():
    from rait.biort_sys import biortdc_generate
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    result = biortdc_generate(mpoles, coeffs)
    expected_result = torch.tensor([3.200000+-0.600000j,-2.171179+3.157642j,-4.628821+-1.757640j,3.199998+-0.600003j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #input validation

    #mpoles is already tested with check_poles(mpoles) in rat_sys.py
    #coeffs
    with pytest.raises(TypeError, match="coeffs must be a torch.Tensor."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = [2j, 0, -2]
        biortdc_generate(mpoles, coeffs)
    with pytest.raises(ValueError, match="coeffs must be a 1-dimensional torch.Tensor."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        biortdc_generate(mpoles, coeffs)
    with pytest.raises(TypeError, match="coeffs must be a complex tensor."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2, 0, -2], dtype=torch.float32)
        biortdc_generate(mpoles, coeffs)
    #coeffs and poles do not have the same length
    with pytest.raises(ValueError, match="mpoles and coeffs must have the same number of elements."):
        mpoles = torch.tensor([-0.5j, 0], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        biortdc_generate(mpoles, coeffs)
    #eps
    with pytest.raises(TypeError, match="eps must be a float."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        biortdc_generate(mpoles, coeffs, eps=1)
    with pytest.raises(ValueError, match="eps must be a positive float."):
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        biortdc_generate(mpoles, coeffs, eps=0.)
