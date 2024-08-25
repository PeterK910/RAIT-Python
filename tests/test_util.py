import torch
import pytest
import re 

def test_check_poles():
    from rait.util import check_poles
    #a
    regex = re.compile(re.escape('poles must be a torch.Tensor.'))
    with pytest.raises(TypeError, match=regex):
        poles = [1, 2, 3]
        check_poles(poles)

    regex = re.compile(re.escape('poles must be complex numbers.'))
    with pytest.raises(TypeError, match=regex):
        poles = torch.tensor([1.0, 2.0, 3.0])
        check_poles(poles)

    regex = re.compile(re.escape('poles must be a 1-dimensional torch.Tensor.'))
    with pytest.raises(ValueError, match=regex):
        poles = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=torch.complex64)
        check_poles(poles)

    regex = re.compile(re.escape('poles must be inside the unit circle!'))
    with pytest.raises(ValueError, match=regex):
        poles = torch.tensor([0.5, 0.5, 1], dtype=torch.complex64)
        check_poles(poles)

#addimag test
def test_addimag():
    from rait.util import addimag
    v1 = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(addimag(v1), torch.tensor([2.+0.j, 2.+0.j, 2.+0.j]))
    v2 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert torch.equal(addimag(v2), torch.tensor([1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]))
    v3 = torch.tensor([-1.0, 0.0, 1.0, 4.0])
    assert torch.equal(addimag(v3), torch.tensor([ 0.+2.j, -1.-1.j,  2.-2.j,  3.+1.j]))

    #input validation
    with pytest.raises(TypeError, match="v must be a torch.Tensor."):
        v4 = [1, 2, 3]
        addimag(v4)
    with pytest.raises(TypeError, match="v must have real elements."):
        v5 = torch.tensor([1+0j, 2, 3])
        addimag(v5)
    with pytest.raises(ValueError, match="v must be a 1-dimensional torch.Tensor."):
        v6 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        addimag(v6)

def test_bisection_order():
    from rait.util import bisection_order
    result = bisection_order(4)
    expected_result = torch.tensor(
        [[ 0, -1, -1],
        [ 4, -1, -1],
        [ 2,  0,  4],
        [ 1,  0,  2],
        [ 3,  2,  4]], dtype=torch.int32)
    assert torch.equal(result, expected_result)
    #input validation
    with pytest.raises(ValueError, match="n must be a non-negative integer."):
        bisection_order(5.3)
    with pytest.raises(ValueError, match="n must be a non-negative integer."):
        bisection_order(-1)

def test_multiplicity():
    from rait.util import multiplicity

    a = torch.tensor([0,0,0.5,0,-0.5j], dtype=torch.complex64)
    spoles, mult = multiplicity(a)
    expected_spoles = torch.tensor([0, 0.5, -0.5j], dtype=torch.complex64)
    expected_mult = torch.tensor([3, 1, 1], dtype=torch.int32)
    assert torch.allclose(spoles, expected_spoles)
    assert torch.equal(mult, expected_mult)

def test_kernel():
    from rait.util import kernel
    #TODO: type specification of y and z
    #when y and z are equal
    y=torch.tensor(-1j, dtype=torch.complex64)
    z=torch.tensor(-1j, dtype=torch.complex64)
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor(4.6000+0.j,dtype=torch.complex64)
    assert torch.allclose(kernel(y, z, mpoles), expected_result)

    #when y and z are different
    y=torch.tensor(-1j, dtype=torch.complex64)
    z=torch.tensor(0.5, dtype=torch.complex64)
    mpoles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor(0.8000-0.4000j,dtype=torch.complex64)

def test_discretize_dc():
    from rait.util import discretize_dc

    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor([-3.141593, -1.220906,  0.293611,  3.141591], dtype=torch.float64)
    assert torch.allclose(discretize_dc(a), expected_result)

    #input validation
    #a is already tested with check_poles(a) in util.py, so done here

def test_discretize_dr():
    from rait.util import discretize_dr

    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor([-2.547801, -1.727318, -1.088921, -0.372949,  0.261651,  1.153941, 2.466805], dtype=torch.float64)
    assert torch.allclose(discretize_dr(a), expected_result)

    #input validation
    #a is already tested with check_poles(a) in util.py, so done here

def test_subsample():
    from rait.util import subsample, discretize_dr
    sample = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    x = discretize_dr(sample)
    expected_result=torch.tensor([0.000000-0.405495j, 0.000000-0.274911j, 0.000000-0.173307j, 0.000000-0.059357j, 0.041643+0.000000j, 0.183655+0.000000j, 0.392604+0.000000j], dtype=torch.complex128)
    assert torch.allclose(subsample(sample, x), expected_result)

    #input validation
    #sample
    with pytest.raises(TypeError, match="sample must be a torch.Tensor."):
        sample = [1, 2, 3]
        subsample(sample, x)
    with pytest.raises(ValueError, match="sample must be a 1-dimensional torch.Tensor."):
        sample = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        subsample(sample, x)
    with pytest.raises(TypeError, match="sample must have complex elements."):
        sample = torch.tensor([1.0, 2.0, 3.0])
        subsample(sample, x)
    #x
    with pytest.raises(TypeError, match="x must be a torch.Tensor."):
        sample = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        x = [1, 2, 3]
        subsample(sample, x)
    with pytest.raises(ValueError, match="x must be a 1-dimensional torch.Tensor"):
        sample = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        subsample(sample, x)
    with pytest.raises(TypeError, match="x must have real elements."):
        sample = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        x = torch.tensor([1.0+1j, 2.0, 3.0])
        subsample(sample, x)

def test_dotdc():
    from rait.util import dotdc
    F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
    G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    t = torch.tensor([1, 2, 3,4], dtype=torch.float32)
    expected_result = torch.tensor(59.374687j, dtype=torch.complex64)
    assert torch.allclose(dotdc(F, G, poles, t), expected_result)

    #input validation
    #F
    with pytest.raises(TypeError, match="F must be a torch.Tensor."):
        F = [1, 2, 3]
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3,4], dtype=torch.float32)
        dotdc(F, G, poles, t)
    with pytest.raises(ValueError, match="F must be a 1-dimensional torch.Tensor."):
        F = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3,4], dtype=torch.float32)
        dotdc(F, G, poles, t)
    with pytest.raises(TypeError, match="F must have complex elements."):
        F = torch.tensor([1.0, 2.0, 3.0])
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3,4], dtype=torch.float32)
        dotdc(F, G, poles, t)
    #G
    with pytest.raises(TypeError, match="G must be a torch.Tensor."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = [1, 2, 3]
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3,4], dtype=torch.float32)
        dotdc(F, G, poles, t)
    with pytest.raises(ValueError, match="G must be a 1-dimensional torch.Tensor."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3,4], dtype=torch.float32)
        dotdc(F, G, poles, t)
    with pytest.raises(TypeError, match="G must have complex elements."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([1.0, 2.0, 3.0])
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3,4], dtype=torch.float32)
        dotdc(F, G, poles, t)
    #F and G have different lengths
    with pytest.raises(ValueError, match="F and G must have the same length."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi, 0], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3,4], dtype=torch.float32)
        dotdc(F, G, poles, t)
    #poles is already tested with check_poles(poles) in util.py, so done here
    #t
    with pytest.raises(TypeError, match="t must be a torch.Tensor."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = [1, 2, 3,4]
        dotdc(F, G, poles, t)
    with pytest.raises(ValueError, match="t must be a 1-dimensional torch.Tensor."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([[1, 2, 3,4], [5, 6, 7, 8]])
        dotdc(F, G, poles, t)
    with pytest.raises(TypeError, match="t must have real elements."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1.0j, 2.0, 3.0, 4.0])
        dotdc(F, G, poles, t)

def test_dotdr():
    from rait.util import dotdr
    F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
    G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    t = torch.tensor([1, 2], dtype=torch.float32)
    expected_result = torch.tensor(0.+8.918623j, dtype=torch.complex64)
    assert torch.allclose(dotdr(F, G, poles, t), expected_result)

    #input validation
    #F
    with pytest.raises(TypeError, match="F must be a torch.Tensor."):
        F = [1, 2, 3]
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2], dtype=torch.float32)
        dotdr(F, G, poles, t)
    with pytest.raises(ValueError, match="F must be a 1-dimensional torch.Tensor."):
        F = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2], dtype=torch.float32)
        dotdr(F, G, poles, t)
    with pytest.raises(TypeError, match="F must have complex elements."):
        F = torch.tensor([1.0, 2.0, 3.0])
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2], dtype=torch.float32)
        dotdr(F, G, poles, t)
    #G
    with pytest.raises(TypeError, match="G must be a torch.Tensor."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = [1, 2, 3]
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2], dtype=torch.float32)
        dotdr(F, G, poles, t)
    with pytest.raises(ValueError, match="G must be a 1-dimensional torch.Tensor."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2], dtype=torch.float32)
        dotdr(F, G, poles, t)
    with pytest.raises(TypeError, match="G must have complex elements."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([1.0, 2.0, 3.0])
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2], dtype=torch.float32)
        dotdr(F, G, poles, t)
   
    #mpoles is already tested with check_poles(mpoles) in util.py, so done here
    #t
    with pytest.raises(TypeError, match="t must be a torch.Tensor."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = [1, 2, 3,4]
        dotdr(F, G, poles, t)
    with pytest.raises(ValueError, match="t must be a 1-dimensional torch.Tensor."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([[1, 2, 3,4], [5, 6, 7, 8]])
        dotdr(F, G, poles, t)
    with pytest.raises(TypeError, match="t must have real elements."):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1.0j, 2.0, 3.0, 4.0])
        dotdr(F, G, poles, t)

    #F and G and t have different lengths
    #F differs from all
    with pytest.raises(ValueError, match='F, G, and t must have the same length.'):
        F = torch.tensor([2*torch.pi, torch.pi,0], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2], dtype=torch.float32)
        dotdr(F, G, poles, t)
    #G differs from all
    with pytest.raises(ValueError, match='F, G, and t must have the same length.'):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi, 0], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2], dtype=torch.float32)
        dotdr(F, G, poles, t)
    #t differs from all
    with pytest.raises(ValueError, match='F, G, and t must have the same length.'):
        F = torch.tensor([2*torch.pi, torch.pi], dtype=torch.complex64)
        G = torch.tensor([-2j*torch.pi, -1j*torch.pi], dtype=torch.complex64)
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        t = torch.tensor([1, 2, 3], dtype=torch.float32)
        dotdr(F, G, poles, t)

def test_coeff_conv():
    from rait.util import coeff_conv
    #lf to biort
    length=5
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
    base1='lf'
    base2='biort'
    result = coeff_conv(length, poles, coeffs, base1, base2)
    expected_result = torch.tensor([5.047004-1.681930j,4.600615-1.787050j,4.447637-1.770824j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)
    #mt to mlf
    length=5
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
    base1='mt'
    base2='mlf'
    result = coeff_conv(length, poles, coeffs, base1, base2)
    expected_result = torch.tensor([5.196152+-0.401924j,-6.928203+4.000000j,4.330127+-2.598076j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #input validation
    #length
    with pytest.raises(TypeError, match="length must be an integer."):
        length=5.3
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='lf'
        base2='biort'
        coeff_conv(length, poles, coeffs, base1, base2)
    with pytest.raises(ValueError, match="length must be an integer greater than or equal to 2."):
        length=1
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='lf'
        base2='biort'
        coeff_conv(length, poles, coeffs, base1, base2)
    #poles is already tested with check_poles(poles) in util.py, so done here
    #coeffs
    with pytest.raises(TypeError, match="coeffs must be a torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = [3, 2, -2j]
        base1='lf'
        base2='biort'
        coeff_conv(length, poles, coeffs, base1, base2)
    with pytest.raises(ValueError, match="coeffs must be a 1-dimensional torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([[3, 2, -2j], [3, 2, -2j]])
        base1='lf'
        base2='biort'
        coeff_conv(length, poles, coeffs, base1, base2)
    with pytest.raises(TypeError, match="coeffs must have complex elements."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, 3], dtype=torch.float64)
        base1='lf'
        base2='biort'
        coeff_conv(length, poles, coeffs, base1, base2)
    #coeffs and poles have different lengths
    with pytest.raises(ValueError, match="coeffs must have the same length as poles."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2], dtype=torch.complex64)
        base1='lf'
        base2='biort'
        coeff_conv(length, poles, coeffs, base1, base2)
    #base1
    with pytest.raises(TypeError, match="base1 must be a string."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1=3
        base2='biort'
        coeff_conv(length, poles, coeffs, base1, base2)
    with pytest.raises(ValueError, match="Invalid system type for base1! Choose from lf, mlf, biort, mt."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='lf1'
        base2='biort'
        coeff_conv(length, poles, coeffs, base1, base2)
    #base2
    with pytest.raises(TypeError, match="base2 must be a string."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='lf'
        base2=3
        coeff_conv(length, poles, coeffs, base1, base2)
    with pytest.raises(ValueError, match="Invalid system type for base2! Choose from lf, mlf, biort, mt."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='lf'
        base2='biort1'
        coeff_conv(length, poles, coeffs, base1, base2)

def test_coeffd_conv():
    from rait.util import coeffd_conv

    #mlfdc to biortdc
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
    base1='mlfdc'
    base2='biortdc'
    result = coeffd_conv(poles, coeffs, base1, base2)
    expected_result = torch.tensor([5.529412+-1.882353j,5.000000+-2.000000j,4.823529+-1.960784j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #mtdc to mlfdc
    poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
    base1='mtdc'
    base2='mlfdc'
    result = coeffd_conv(poles, coeffs, base1, base2)
    expected_result = torch.tensor([5.196152+-0.401924j,-6.928203+4.000000j,4.330127+-2.598076j], dtype=torch.complex64)
    assert torch.allclose(result, expected_result)

    #input validation
    #poles is already tested with check_poles(poles) in util.py, so done here
    #coeffs
    with pytest.raises(TypeError, match="coeffs must be a torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = [3, 2, -2j]
        base1='mtdc'
        base2='mlfdc'
        coeffd_conv(poles, coeffs, base1, base2)
    with pytest.raises(ValueError, match="coeffs must be a 1-dimensional torch.Tensor."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([[3, 2, -2j], [3, 2, -2j]])
        base1='mtdc'
        base2='mlfdc'
        coeffd_conv(poles, coeffs, base1, base2)
    with pytest.raises(TypeError, match="coeffs must have complex elements."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, 3], dtype=torch.float64)
        base1='mtdc'
        base2='mlfdc'
        coeffd_conv(poles, coeffs, base1, base2)
    #coeffs and poles have different lengths
    with pytest.raises(ValueError, match="coeffs must have the same length as poles."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2], dtype=torch.complex64)
        base1='mtdc'
        base2='mlfdc'
        coeffd_conv(poles, coeffs, base1, base2)
    #base1
    with pytest.raises(TypeError, match="base1 must be a string."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1=3
        base2='mlfdc'
        coeffd_conv(poles, coeffs, base1, base2)
    with pytest.raises(ValueError, match="Invalid system type for base1! Choose from mlfdc, biortdc, mtdc."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='mtdc1'
        base2='mlfdc'
        coeffd_conv(poles, coeffs, base1, base2)
    #base2
    with pytest.raises(TypeError, match="base2 must be a string."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='mtdc'
        base2=3
        coeffd_conv(poles, coeffs, base1, base2)
    with pytest.raises(ValueError, match="Invalid system type for base2! Choose from mlfdc, biortdc, mtdc."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='mtdc'
        base2='mlfdc1'
        coeffd_conv(poles, coeffs, base1, base2)
    #eps
    with pytest.raises(TypeError, match="eps must be a float."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='mtdc'
        base2='mlfdc'
        eps='0.1'
        coeffd_conv(poles, coeffs, base1, base2, eps)
    with pytest.raises(ValueError, match="eps must be a positive float."):
        length=5
        poles = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
        coeffs = torch.tensor([3, 2, -2j], dtype=torch.complex64)
        base1='mtdc'
        base2='mlfdc'
        eps=0.
        coeffd_conv(poles, coeffs, base1, base2, eps)
