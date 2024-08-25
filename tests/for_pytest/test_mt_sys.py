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

def test_mtdr_system():
    from .mt_sys import mtdr_system
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_re = torch.tensor(
        [[1.000000,1.000000,1.000000,1.000000,1.000000,1.000000,1.000000],
        [-1.039524,-0.514822,1.103002,0.910635,0.554493,0.162007,-0.360706],
        [0.989743,-0.814665,0.180513,0.958454,0.236961,-0.848516,-0.349550],
        [-0.514604,0.682016,-0.966893,1.405432,-0.429582,-0.413126,0.440526]], dtype=torch.float64)
    expected_im = torch.tensor(
        [[0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000],
        [-0.074635,-1.610932,-0.918993,0.132633,0.435503,0.565929,0.519566],
        [0.142858,0.579932,-0.983573,0.285247,0.971519,0.529170,-0.936918],
        [-0.309780,0.261393,-0.136528,-0.614615,1.567154,-0.846630,0.418619],], dtype=torch.float64)
    re, im = mtdr_system(poles)
    assert torch.allclose(re, expected_re)
    assert torch.allclose(im, expected_im)

    #input validation
    #poles is already tested with check_poles(poles) in mt_sys.py
    with pytest.raises(TypeError, match="eps must be a float"):
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdr_system(poles, eps=1)
    with pytest.raises(ValueError, match="eps must be positive"):
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdr_system(poles, eps=-1.)
    with pytest.raises(ValueError, match="eps must be positive"):
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdr_system(poles, eps=0.)

def test_mt_coeffs():
    from .mt_sys import mt_coeffs
    v = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_co = torch.tensor([-0.621933+0.530092j,0.415187+0.256114j,0.348194+0.349086j], dtype=torch.complex64)
    expected_err = torch.tensor(1.767295)
    co,err = mt_coeffs(v, poles)
    err = torch.tensor(err)
    assert torch.allclose(co, expected_co)
    assert torch.allclose(err, expected_err)

    #input validation
    with pytest.raises(TypeError, match="v must be a torch.Tensor."):
        v = [2j, 0, -2]
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mt_coeffs(v, poles)
    with pytest.raises(ValueError, match="v must be a 1-dimensional torch.Tensor."):
        v = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mt_coeffs(v, poles)
    with pytest.raises(TypeError, match="v must be a complex tensor."):
        v = torch.tensor([2, 0, -2], dtype=torch.float32)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mt_coeffs(v, poles)
    #poles is already tested with check_poles(poles) in mt_sys.py

def test_mtdc_coeffs():
    from .mt_sys import mtdc_coeffs
    signal = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    co,err = mtdc_coeffs(signal, poles)
    err = torch.tensor(err)
    expected_co = torch.tensor([-0.300508+1.000102j,0.703239+-0.428161j,-0.592564+0.317047j], dtype=torch.complex64)
    expected_err = torch.tensor(3.080572)
    assert torch.allclose(co, expected_co)
    assert torch.allclose(err, expected_err)

    #input validation
    with pytest.raises(TypeError, match="signal must be a torch.Tensor."):
        signal = [2j, 0, -2]
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdc_coeffs(signal, mpoles)
    with pytest.raises(ValueError, match="signal must be a 1-dimensional torch.Tensor."):
        signal = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdc_coeffs(signal, mpoles)
    with pytest.raises(TypeError, match="signal must be a complex tensor."):
        signal = torch.tensor([2, 0, -2], dtype=torch.float32)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdc_coeffs(signal, mpoles)
    #mpoles is already tested with check_poles(mpoles) in rat_sys.py
    #eps
    with pytest.raises(TypeError, match="eps must be a float."):
        signal = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdc_coeffs(signal, mpoles, eps=1)
    with pytest.raises(ValueError, match="eps must be a positive float."):
        signal = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdc_coeffs(signal, mpoles, eps=0.)

def test_mtdr_generate():
    from .mt_sys import mtdr_generate

    length = 5
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    cUk = torch.tensor([1,2,3,4], dtype=torch.float64)
    cVk = torch.tensor([-1,-2,-3,-4], dtype=torch.float64)
    result = mtdr_generate(length, mpoles, cUk, cVk)
    expected_result= torch.tensor([-1.223245,1.949876,25.660724,-19.231660,2.595392], dtype=torch.float32)
    assert torch.allclose(result, expected_result)

    #input validation
    #length
    with pytest.raises(TypeError, match="length must be an integer."):
        length = 5.3
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        cUk = torch.tensor([1,2,3,4], dtype=torch.float64)
        cVk = torch.tensor([-1,-2,-3,-4], dtype=torch.float64)
        mtdr_generate(length, mpoles, cUk, cVk)
    with pytest.raises(ValueError, match="length must be an integer greater than or equal to 2."):
        length = 1
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        cUk = torch.tensor([1,2,3,4], dtype=torch.float64)
        cVk = torch.tensor([-1,-2,-3,-4], dtype=torch.float64)
        mtdr_generate(length, mpoles, cUk, cVk)
    #mpoles is already tested with check_poles(mpoles) in mt_sys.py
    #cUk
    with pytest.raises(TypeError, match="cUk must be a torch.Tensor."):
        length = 5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        cUk = [1,2,3,4]
        cVk = torch.tensor([-1,-2,-3,-4], dtype=torch.float64)
        mtdr_generate(length, mpoles, cUk, cVk)
    with pytest.raises(ValueError, match="cUk must be a 1-dimensional torch.Tensor."):
        length = 5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        cUk = torch.tensor([[1,2,3,4], [1,2,3,4]], dtype=torch.float64)
        cVk = torch.tensor([-1,-2,-3,-4], dtype=torch.float64)
        mtdr_generate(length, mpoles, cUk, cVk)
    with pytest.raises(TypeError, match="cUk must be a float tensor."):
        length = 5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        cUk = torch.tensor([1,2,3,4], dtype=torch.int64)
        cVk = torch.tensor([-1,-2,-3,-4], dtype=torch.float64)
        mtdr_generate(length, mpoles, cUk, cVk)
    #cVk
    with pytest.raises(TypeError, match="cVk must be a torch.Tensor."):
        length = 5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        cUk = torch.tensor([1,2,3,4], dtype=torch.float64)
        cVk = [-1,-2,-3,-4]
        mtdr_generate(length, mpoles, cUk, cVk)
    with pytest.raises(ValueError, match="cVk must be a 1-dimensional torch.Tensor."):
        length = 5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        cUk = torch.tensor([1,2,3,4], dtype=torch.float64)
        cVk = torch.tensor([[-1,-2,-3,-4], [-1,-2,-3,-4]], dtype=torch.float64)
        mtdr_generate(length, mpoles, cUk, cVk)
    with pytest.raises(TypeError, match="cVk must be a float tensor."):
        length = 5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        cUk = torch.tensor([1,2,3,4], dtype=torch.float64)
        cVk = torch.tensor([-1,-2,-3,-4], dtype=torch.int64)
        mtdr_generate(length, mpoles, cUk, cVk)
    
    #length of mpoles not equal to length(cUk)-1
    with pytest.raises(ValueError, match="mpoles must have 1 less element than cUk."):
        length = 5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        cUk = torch.tensor([1,2,3], dtype=torch.float64)
        cVk = torch.tensor([-1,-2,-3,-4], dtype=torch.float64)
        mtdr_generate(length, mpoles, cUk, cVk)
    #length of mpoles not equal to length(cVk)-1
    with pytest.raises(ValueError, match="mpoles must have 1 less element than cVk."):
        length = 5
        mpoles = torch.tensor([-0.5j, 0,0.5], dtype=torch.complex64)
        cUk = torch.tensor([1,2,3,4], dtype=torch.float64)
        cVk = torch.tensor([-1,-2,-3,-4,-5], dtype=torch.float64)
        mtdr_generate(length, mpoles, cUk, cVk)

def test_mtdr_coeffs():
    from .mt_sys import mtdr_coeffs

    v = torch.tensor([2, 0, -1], dtype=torch.float64)
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    cUk, cVk, err = mtdr_coeffs(v, mpoles)
    err = torch.tensor(err)
    expected_cUk = torch.tensor([0.240428,-0.189690,0.334254,-0.148325], dtype=torch.float64)
    expected_cVk = torch.tensor([0.000000,-0.400554,0.153981,-0.111509], dtype=torch.float64)
    expected_err = torch.tensor(1.935322)
    assert torch.allclose(cUk, expected_cUk)
    assert torch.allclose(cVk, expected_cVk)
    assert torch.allclose(err, expected_err)

    #input validation
    #v
    with pytest.raises(TypeError, match="v must be a torch.Tensor."):
        v = [2, 0, -1]
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdr_coeffs(v, mpoles)
    with pytest.raises(ValueError, match="v must be a 1-dimensional torch.Tensor."):
        v = torch.tensor([[2, 0, -1], [2, 0, -1]], dtype=torch.float64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdr_coeffs(v, mpoles)
    with pytest.raises(TypeError, match="v must be a float tensor."):
        v = torch.tensor([2, 0, -1], dtype=torch.int64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdr_coeffs(v, mpoles)
    #mpoles is already tested with check_poles(mpoles) in mt_sys.py
    #eps
    with pytest.raises(TypeError, match="eps must be a float."):
        v = torch.tensor([2, 0, -1], dtype=torch.float64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdr_coeffs(v, mpoles, eps=1)
    with pytest.raises(ValueError, match="eps must be a positive float."):
        v = torch.tensor([2, 0, -1], dtype=torch.float64)
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        mtdr_coeffs(v, mpoles, eps=0.)

def test_mt_generate():
    from .mt_sys import mt_generate
    length=5
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    result = mt_generate(length, poles, coeffs)
    expected_result = torch.tensor([-0.000000+2.309401j,0.865771+1.651865j,-3.094895+3.475989j,1.397230+-0.688440j,1.077684+2.166189j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #input validation
    #length
    with pytest.raises(TypeError, match="length must be an integer."):
        length=2.3
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mt_generate(length, poles, coeffs)
    with pytest.raises(ValueError, match="length must be greater than or equal to 2."):
        length=1
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mt_generate(length, poles, coeffs)
    #poles is already tested with check_poles(poles) in rat_sys.py
    #coeffs
    with pytest.raises(TypeError, match="coeffs must be a torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = [2j, 0, -2]
        mt_generate(length, poles, coeffs)
    with pytest.raises(ValueError, match="coeffs must be a 1-dimensional torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        mt_generate(length, poles, coeffs)
    with pytest.raises(TypeError, match="coeffs must be a complex tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2, 0, -2], dtype=torch.float32)
        mt_generate(length, poles, coeffs)
    #coeffs and poles do not have the same length
    with pytest.raises(ValueError, match="poles and coeffs must have the same number of elements."):
        length=5
        poles = torch.tensor([-0.5j, 0], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mt_generate(length, poles, coeffs)

def test_mtdc_generate():
    from .mt_sys import mtdc_generate
    length=5
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
    result = mtdc_generate(length, mpoles, coeffs)
    expected_result = torch.tensor([-0.000000+2.309401j,0.865771+1.651865j,-3.094895+3.475989j,1.397230+-0.688440j,1.077684+2.166189j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #input validation
    #length
    with pytest.raises(TypeError, match="length must be an integer."):
        length=2.3
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mtdc_generate(length, mpoles, coeffs)
    with pytest.raises(ValueError, match="length must be greater than or equal to 2."):
        length=1
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mtdc_generate(length, mpoles, coeffs)
    #poles is already tested with check_poles(poles) in rat_sys.py
    #coeffs
    with pytest.raises(TypeError, match="coeffs must be a torch.Tensor."):
        length=5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = [2j, 0, -2]
        mtdc_generate(length, mpoles, coeffs)
    with pytest.raises(ValueError, match="coeffs must be a 1-dimensional torch.Tensor."):
        length=5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([[2j, 0, -2], [2j, 0, -2]], dtype=torch.complex64)
        mtdc_generate(length, mpoles, coeffs)
    with pytest.raises(TypeError, match="coeffs must be a complex tensor."):
        length=5
        mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([2, 0, -2], dtype=torch.float32)
        mtdc_generate(length, mpoles, coeffs)
    #coeffs and poles do not have the same length
    with pytest.raises(ValueError, match="mpoles and coeffs must have the same number of elements."):
        length=5
        mpoles = torch.tensor([-0.5j, 0], dtype=torch.complex64)
        coeffs = torch.tensor([2j, 0, -2], dtype=torch.complex64)
        mtdc_generate(length, mpoles, coeffs)