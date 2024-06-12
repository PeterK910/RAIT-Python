import torch
from other import multiplicity, discretize_dc

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

