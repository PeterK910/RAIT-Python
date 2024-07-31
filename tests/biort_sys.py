import torch
from math import factorial
from scipy.special import binom



def biort_system(length:int, mpoles:torch.Tensor) -> torch.Tensor:
    """
    Generates the biorthogonal system.

    The following functions and notations are used from:

    S. Fridli, F. Schipp, Biorthogonal systems to rational functions, 
    Annales Univ. Sci. Budapest., Sect. Comp, vol. 35, no. 1, 
    pp. 95-105, 2011.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling. Must be at least 2.
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the biorthogonal system. Must a 1-dimensional tensor with all its elements inside the unit circle.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The elements of the biorthogonal system at the uniform sampling points as row vectors.
    
    Raises
    ------
    ValueError
        If the number of poles is not 1 or the length is less than 2.
        Also, if the poles are not inside the unit circle.
    """
    from .util import check_poles, multiplicity
    
    if type(length) != int:
        raise TypeError('Length must be an integer.')
    if length < 2:
        raise ValueError('Length must be at least 2.')
    
    check_poles(mpoles)

    bts = torch.zeros((mpoles.numel(), length), dtype=torch.complex64)
    t = torch.linspace(-torch.pi, torch.pi, length + 1)[:-1]
    z = torch.exp(1j * t)

    spoles, multi = multiplicity(mpoles)

    for j in range(len(multi)):
        for k in range(1, multi[j] + 1):
            col = sum(multi[:j]) + k
            bts[col - 1, :] = __pszi(j, k, spoles, multi, z)

    return bts

def __pszi(l:int, j:int, poles:torch.Tensor, multi:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
    """
    Compute the values of the biorthogonal polynomial at z related to the lth
    pole with j multiplicity.

    Parameters
    ----------
    l : int
        The index of the pole.
    j : int
        The multiplicity of the pole.
    poles : torch.Tensor
        The poles of the biorthogonal system.
    multi : torch.Tensor
        The multiplicity of each pole.
    z : torch.Tensor
        The complex plane values where the function is evaluated.

    Returns
    -------
    torch.Tensor
        The calculated values of the biorthogonal polynomial.
    """
    n = len(poles)
    v = torch.zeros(z.size(), dtype=torch.complex64)
    do = __domega(int(multi[l]) - j, l, poles, multi, poles[l])

    for s in range(multi[l] - j + 1):
        v += do[s] / factorial(s) * (z - poles[l]) ** s

    v *= __omega(l, poles, multi, z) / __omega(l, poles, multi, poles[l]) * (z - poles[l]) ** (j - 1)
    return v

def __omega(l:int, poles:torch.Tensor, multi:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
    """
    Computes the values of the Omega base functions related to the
    biorthogonal system.

    Parameters
    ----------
    l : int
        The index of the pole.
    poles : torch.Tensor
        The poles of the biorthogonal system.
    multi : torch.Tensor
        The multiplicity of each pole.
    z : torch.Tensor
        The complex plane values where the function is evaluated.

    Returns
    -------
    torch.Tensor
        The calculated values of the Omega base functions.
    """
    n = len(poles)
    v = torch.ones(z.size(), dtype=torch.complex64)
    v /= (1 - poles[l].conj() * z) ** multi[l]

    # Blaschke-function
    def B(z, a):
        return (z - a) / (1 - a.conj() * z)

    for i in range(l):
        v *= B(z, poles[i]) ** multi[i]
    for i in range(l + 1, n):
        v *= B(z, poles[i]) ** multi[i]

    return v

def __domega(s:int, l:int, poles:torch.Tensor, multi:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
    """
    Computes sth derivative of the omega function.

    The first row of array 'Do' contains the values of omega.

    The rth derivative is stored in Do(r+1,:). 

    Parameters
    ----------
    s : int
        The order of the derivative.
    l : int
        The index of the pole.
    poles : torch.Tensor
        The poles of the biorthogonal system.
    multi : torch.Tensor
        The multiplicity of each pole.
    z : torch.Tensor
        The complex plane values where the function is evaluated.

    Returns
    -------
    torch.Tensor
        The calculated values of the sth derivative of the omega function.
    """
    n = len(poles)
    Do = torch.zeros((s + 1, z.numel()), dtype=torch.complex64)
    Do[0, :] = __omega(l, poles, multi, poles[l]) / __omega(l, poles, multi, z)

    for i in range(1, s + 1):
        for j in range(1, i + 1):
            Do[i, :] += binom(i - 1, j - 1) * Do[j - 1, :] * __ro(i - j, l, poles, multi, z)

    return Do

def __ro(s:int, l:int, poles:torch.Tensor, multi:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
    """
    Computes sth derivative of the auxiliary function to Domega.

    Parameters
    ----------
    s : int
        The order of the derivative.
    l : int
        The index of the pole.
    poles : torch.Tensor
        The poles of the biorthogonal system.
    multi : torch.Tensor
        The multiplicity of each pole.
    z : torch.Tensor
        The complex plane values where the function is evaluated.

    Returns
    -------
    torch.Tensor
        The calculated values of the sth derivative of the auxiliary function.
    """
    n = len(multi)
    v = torch.ones(z.size(), dtype=torch.cfloat)
    v *= multi[l] / (z - (1 / poles[l].conj())) ** (s + 1)

    for i in range(l):
        v -= multi[i] * ((1 / (z - poles[i])) ** (s + 1) - (1 / (z - (1 / poles[i].conj()))) ** (s + 1))
    for i in range(l, n):
        v -= multi[i] * ((1 / (z - poles[i])) ** (s + 1) - (1 / (z - (1 / poles[i].conj()))) ** (s + 1))

    v *= (-1) ** s * factorial(s)
    return v



def biortdc_system(mpoles:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """
    Generates the discrete biorthogonal system.

    Parameters
    ----------
    mpoles : torch.Tensor
        Poles of the discrete biorthogonal system.
    eps : float, optional
        Accuracy of the discretization on the unit disc (default is 1e-6).

    Returns
    -------
    torch.Tensor
        The elements of the discrete biorthogonal system at the uniform
        sampling points as row vectors.

    Raises
    ------
    ValueError
        If the poles are not inside the unit circle.
    """
    from util import discretize_dc, multiplicity

    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError('Poles must be inside the unit circle!')
    
    m = mpoles.numel()
    bts = torch.zeros(m, m+1, dtype=mpoles.dtype)
    t = discretize_dc(mpoles, eps)

    spoles, multi = multiplicity(mpoles)

    for j in range(len(multi)):
        for k in range(1, multi[j]+1):
            col = sum(multi[:j]) + k
            bts[col-1, :] = __pszi(j+1, k, spoles, multi, torch.exp(1j * t))

    return bts



def biort_coeffs(v: torch.Tensor, poles: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Calculate the biorthogonal-coefficients of 'v' with respect to the 
    biorthogonal system given by 'poles'.

    Parameters
    ----------
    v : torch.Tensor
        An arbitrary vector.
    poles : torch.Tensor
        Poles of the biorthogonal system.

    Returns
    -------
    torch.Tensor
        The Fourier coefficients of v with respect to the biorthogonal 
        system defined by poles.
    float
        L^2 norm of the approximation error.
    
    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from rat_sys import mlf_system

    # Validate input parameters
    if not isinstance(v, torch.Tensor) or v.ndim != 1:
        raise ValueError('v must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(poles, torch.Tensor) or poles.ndim != 1:
        raise ValueError('Poles must be a 1-dimensional torch.Tensor.')
    
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must be inside the unit circle!')
    
    # Calculate the biorthogonal system elements
    mlfs = mlf_system(v.size(0), poles)
    bts = biort_system(v.size(0), poles)
    
    # Calculate coefficients and error
    co = (mlfs @ v.unsqueeze(1) / v.size(0)).squeeze()
    err = torch.linalg.norm(co @ bts - v).item()
    
    return co, err

def biort_generate(length: int, poles: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Generates a function in the space spanned by the biorthogonal system.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor
        Poles of the biorthogonal system (row vector).
    coeffs : torch.Tensor
        Coefficients of the linear combination to form (row vector).

    Returns
    -------
    torch.Tensor
        The generated function at the uniform sampling points as a row vector.
    
        It is the linear combination of the LF system elements.
        
    Table
    -----
    +-------+-------+-------+-------+-------+-------+-------+
    |       |       |  bt1  |  bt1  |  bt1  | ...   |  bt1  |
    |       |       |  bt2  |  bt2  |  bt2  | ...   |  bt2  |
    | co1   | co2   |   v   |   v   |   v   | ...   |   v   |
    +-------+-------+-------+-------+-------+-------+-------+

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    
    if not isinstance(length, int) or length < 2:
        raise ValueError("Length must be an integer greater than or equal to 2.")
    
    if not isinstance(poles, torch.Tensor) or poles.ndim != 1:
        raise ValueError("Poles must be a 1-dimensional torch.Tensor.")
    
    if not isinstance(coeffs, torch.Tensor) or coeffs.ndim != 1:
        raise ValueError("Coeffs must be a 1-dimensional torch.Tensor.")
    
    if poles.shape[0] != coeffs.shape[0]:
        raise ValueError("Poles and coeffs must have the same number of elements.")
    
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError("Poles must be inside the unit circle!")
    
    # Calculate the biorthogonal system elements
    v = coeffs @ biort_system(length, poles)
    
    return v

def biortdc_generate(length: int, mpoles: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Generates a function in the space spanned by the discrete biorthogonal system.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the biorthogonal system (1-dimensional tensor).
    coeffs : torch.Tensor, dtype=torch.complex64
    TODO: coeffs dtype?
        Coefficients of the linear combination to form (1-dimensional tensor).

    Returns
    -------
    torch.Tensor
        The generated function at the uniform sampling points as a 1-dimensional tensor.

        It is the linear combination of the discrete biorthogonal system elements.

    Table
    -----
    +-------+-------+-------+-------+-------+-------+-------+
    |       |       | btdc1 | btdc1 | btdc1 | ...   | btdc1 |
    |       |       | btdc2 | btdc2 | btdc2 | ...   | btdc2 |
    | co1   | co2   |   v   |   v   |   v   | ...   |   v   |
    +-------+-------+-------+-------+-------+-------+-------+

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(length, int):
        raise TypeError('length must be an integer.')
    if length < 2:
        raise ValueError('length must be at least 2.')

    
    if not isinstance(mpoles, torch.Tensor) or mpoles.ndim != 1:
        raise ValueError('mpoles must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(coeffs, torch.Tensor) or coeffs.ndim != 1:
        raise ValueError('coeffs must be a 1-dimensional torch.Tensor.')
    
    if mpoles.shape[0] != coeffs.shape[0]:
        raise ValueError('mpoles and coeffs must have the same number of elements.')
    
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError('mpoles must be inside the unit circle!')

    # Calculate the biorthogonal system elements
    bts = biort_system(length, mpoles)
    
    # Calculate the linear combination of the discrete biorthogonal system elements
    v = coeffs @ bts

    return v



def biortdc_coeffs(v: torch.Tensor, mpoles: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, float]:
    """
    Calculate the discrete biorthogonal-coefficients of 'v' with respect to 
    the biorthogonal system given by 'mpoles'.

    Parameters
    ----------
    v : torch.Tensor
        An arbitrary vector.
    mpoles : torch.Tensor
        Poles of the biorthogonal system.
    eps : float, optional
        Accuracy of the discretization on the unit disc. Default is 1e-6.

    Returns
    -------
    torch.Tensor
        The Fourier coefficients of v with respect to the discrete 
        biorthogonal system defined by 'mpoles'.
    float
        L^2 norm of the approximation error.
    
    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from util import discretize_dc, subsample, dotdc
    from rat_sys import mlfdc_system
    # Validate input parameters
    if not isinstance(v, torch.Tensor) or v.ndim != 1:
        raise ValueError('v must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(mpoles, torch.Tensor) or mpoles.ndim != 1:
        raise ValueError('mpoles must be a 1-dimensional torch.Tensor.')
    
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError('mpoles must be inside the unit circle!')
    
    if not isinstance(eps, float):
        raise ValueError('eps must be a float.')

    # Discretize and sample
    t = discretize_dc(mpoles, eps)
    samples = subsample(v, t)

    # Calculate coefficients
    m = len(mpoles)
    co = torch.zeros(m)
    mlf = mlfdc_system(mpoles, eps)

    for i in range(m):
        co[i] = dotdc(samples, mlf[i], mpoles, t)

    # Calculate error
    len_v = len(v)
    bts = biort_system(len_v, mpoles)
    err = torch.norm(co @ bts - v).item()

    return co, err

