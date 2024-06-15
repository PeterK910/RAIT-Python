import torch
from other import multiplicity, discretize_dc
from biort_sys import biort_system

def mlf_system(length: int, mpoles: torch.Tensor) -> torch.Tensor:
    """
    Generates the modified basic rational function system defined by mpoles.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    mpoles : torch.Tensor
        Poles of the modified basic rational system.

    Returns
    -------
    torch.Tensor
        The elements of the modified basic rational function system
        at the uniform sampling points as row vectors.

    Raises
    ------
    ValueError
        If the number of poles is not 1 or the length is less than 2.
        Also, if the poles are not inside the unit circle.
    """
    np, mp = mpoles.size()
    if np != 1 or length < 2:
        raise ValueError('Wrong parameters!')
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError('Poles must be inside the unit disc!')

    mlf = torch.zeros(mp, length)
    t = torch.linspace(-torch.pi, torch.pi, length + 1)
    t = t[:-1]
    z = torch.exp(1j * t)

    spoles, multi = multiplicity(mpoles)

    for j in range(len(multi)):
        for k in range(1, multi[j] + 1):
            col = sum(multi[:j]) + k
            mlf[col - 1, :] = torch.pow(z, k - 1) / torch.pow(1 - torch.conj(spoles[j]) * z, k)

    return mlf



def lf_system(length: int, poles: torch.Tensor) -> torch.Tensor:
    """
    Generates the linearly independent system defined by poles.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor
        Poles of the rational system.

    Returns
    -------
    torch.Tensor
        The elements of the linearly independent system at the uniform
        sampling points as row vectors.

    Raises
    ------
    ValueError
        If the number of poles is not 1 or the length is less than 2.
        Also, if the poles are not inside the unit circle.
    """
    np, mp = poles.size()
    if np != 1 or length < 2:
        raise ValueError('Wrong parameters!')
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must be inside the unit disc!')

    lfs = torch.zeros(mp, length, dtype=torch.cfloat) # complex dtype
    t = torch.linspace(-torch.pi, torch.pi, length + 1)[:-1]
    z = torch.exp(1j * t)

    for j in range(mp):
        rec = 1 / (1 - poles[j].conj() * z)
        lfs[j, :] = rec ** __multiplicity_local(j, poles)
        lfs[j, :] /= torch.sqrt(torch.dot(lfs[j, :], lfs[j, :].conj()) / length)

    return lfs

def __multiplicity_local(n:int, v:torch.Tensor) -> int:
    """
    Returns the multiplicity of the nth element of the vector v.

    Parameters
    ----------
    n : int
        The index of the element to check for multiplicity.
    v : torch.Tensor
        The vector containing poles.

    Returns
    -------
    int
        The multiplicity of the nth element.
    """
    m = 0
    for k in range(n):
        if v[k] == v[n]:
            m += 1
    return m

def mlfdc_system(mpoles:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """
    Generates the discrete modified basic rational system.

    Parameters
    ----------
    mpoles : torch.Tensor
        Poles of the discrete modified basic rational system.
    eps : float, optional
        Accuracy of the discretization on the unit disc (default is 1e-6).

    Returns
    -------
    torch.Tensor
        The elements of the discrete modified basic rational system at the
        uniform sampling points as row vectors.

    Raises
    ------
    ValueError
        If the poles are not inside the unit disc.
    """
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError("Poles must be inside the unit disc")

    m = mpoles.numel()
    mlf = torch.zeros(m, m+1, dtype=mpoles.dtype)
    t = discretize_dc(mpoles, eps)

    spoles, multi = multiplicity(mpoles)

    for j in range(len(multi)):
        for k in range(1, multi[j]+1):
            col = sum(multi[:j]) + k
            mlf[col-1, :] = torch.exp(1j * t) ** (k-1) / (1 - spoles[j].conj() * torch.exp(1j * t)) ** k

    return mlf

def mlf_generate(length:int , poles:torch.Tensor, coeffs:torch.Tensor) -> torch.Tensor:
    """
    Compute the values of the poisson function at (r,t).

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor
        Poles of the modified basic rational system (1D tensor).
    coeffs : torch.Tensor
        Coefficients of the linear combination to form (1D tensor).

    Returns
    -------
    torch.Tensor
        The generated function at the uniform sampling points as a 1D tensor.

        It is the linear combination of the discrete real MT system elements.

    Table
    -----
    +-------+-------+-------+-------+-------+-------+-------+
    |       |       | mtdr1 | mtdr1 | mtdr1 | ...   | mtdr1 |
    |       |       | mtdr2 | mtdr2 | mtdr2 | ...   | mtdr2 |
    | co1   | co2   |   v   |   v   |   v   | ...   |   v   |
    +-------+-------+-------+-------+-------+-------+-------+

    Raises
    ------
    ValueError
        If input parameters are incorrect or if poles are outside the valid range.
    """
    
    # Validate input parameters
    if not isinstance(length, int) or length < 2:
        raise ValueError('Parameter `length` must be an integer greater than or equal to 2.')
    
    if not isinstance(poles, torch.Tensor) or poles.dim() != 1:
        raise ValueError('Parameter `poles` must be a 1D torch.Tensor.')
    
    if not isinstance(coeffs, torch.Tensor) or coeffs.dim() != 1 or poles.size(0) != coeffs.size(0):
        raise ValueError('Parameter `coeffs` must be a 1D torch.Tensor with the same size as `poles`.')
    
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must have absolute values less than 1.')

    # Generate the modified rational system elements
    mlf_system_elements = mlf_system(length, poles)

    # Linear combination of the modified rational system elements
    v = torch.matmul(coeffs, mlf_system_elements)

    return v

def mlf_coeffs(v:torch.Tensor, poles:torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Compute the values of the poisson function at (r,t).

    Parameters
    ----------
    v : torch.Tensor
        An arbitrary vector.
    poles : torch.Tensor
        Poles of the modified basic rational system.

    Returns
    -------
    co : torch.Tensor
        The Fourier coefficients of v with respect to the modified basic rational system defined by 'poles'.
    err : float
        L^2 norm of the approximation error.

    Raises
    ------
    ValueError
        If input parameters are incorrect or if poles are outside the valid range.
    """
    
    # Validate input parameters
    if not isinstance(v, torch.Tensor) or v.dim() != 1:
        raise ValueError('Parameter `v` must be a 1D torch.Tensor.')
    
    if not isinstance(poles, torch.Tensor) or poles.dim() != 1:
        raise ValueError('Parameter `poles` must be a 1D torch.Tensor.')
    
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must have absolute values less than 1.')

    # Calculate biorthogonal system elements 
    bts = biort_system(v.size(0), poles)

    # Calculate Fourier coefficients
    co = torch.matmul(bts, v.unsqueeze(1)) / v.size(0)
    
    # Calculate modified rational system elements
    mlf = mlf_system(v.size(0), poles)

    # Calculate approximation error
    err = torch.norm(torch.matmul(co.t(), mlf) - v)

    return co.squeeze(), err.item()

