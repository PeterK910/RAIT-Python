import torch
import pytest
import re 

def test_check_poles():
    from .util import check_poles
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
    from .util import addimag
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
    from .util import bisection_order
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
    from .util import multiplicity

    a = torch.tensor([0,0,0.5,0,-0.5j], dtype=torch.complex64)
    spoles, mult = multiplicity(a)
    expected_spoles = torch.tensor([0, 0.5, -0.5j], dtype=torch.complex64)
    expected_mult = torch.tensor([3, 1, 1], dtype=torch.int32)
    assert torch.allclose(spoles, expected_spoles)
    assert torch.equal(mult, expected_mult)

def test_kernel():
    from .util import kernel
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
    from .util import discretize_dc

    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor([-3.141593, -1.220906,  0.293611,  3.141591], dtype=torch.float64)
    assert torch.allclose(discretize_dc(a), expected_result)

    #input validation
    #a is already tested with check_poles(a) in util.py, so done here

def test_discretize_dr():
    from .util import discretize_dr

    a = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    expected_result = torch.tensor([-2.547801, -1.727318, -1.088921, -0.372949,  0.261651,  1.153941, 2.466805], dtype=torch.float64)
    assert torch.allclose(discretize_dr(a), expected_result)

    #input validation
    #a is already tested with check_poles(a) in util.py, so done here

def test_subsample():
    from .util import subsample, discretize_dr
    sample = torch.tensor([-0.5j, 0, 0.5], dtype=torch.complex64)
    x = discretize_dr(sample)
    expected_result=torch.tensor([[0.000000-0.405495j, 0.000000-0.274911j, 0.000000-0.173307j, 0.000000-0.059357j, 0.041643+0.000000j, 0.183655+0.000000j, 0.392604+0.000000j]], dtype=torch.complex128)
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
    from .util import dotdc
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
    