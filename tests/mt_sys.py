import torch

def mt_system(len: int, poles: torch.Tensor) -> torch.Tensor:
    """
    Generates the Malmquist-Takenaka system.

    Parameters
    ----------
    len : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor
        Poles of the rational system.

    Returns
    -------
    torch.Tensor
        The elements of the MT system at the uniform sampling points as row vectors.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    # Validate input parameters
    if not isinstance(len, int) or len < 2:
        raise ValueError('len must be a positive integer greater than or equal to 2.')

    if not isinstance(poles, torch.Tensor) or poles.ndim != 1:
        raise ValueError('Poles must be a 1-dimensional torch.Tensor.')

    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must be inside the unit circle!')

    # Calculate the MT system elements
    np, mp = poles.size(0), len
    mts = torch.zeros(mp, len)
    t = torch.linspace(-torch.pi, torch.pi, len + 1)[:-1]
    z = torch.exp(1j * t)

    fi = torch.ones(len)  # The product defining MT elements so far
    for j in range(mp):
        co = torch.sqrt(1 - (torch.abs(poles[j]) ** 2))
        rec = 1 / (1 - torch.conj(poles[j]) * z)
        lin = co * rec
        bla = (z - poles[j]) * rec
        mts[j, :] = lin * fi
        fi = fi * bla

    return mts

from .util import discretize_dc

def mtdc_system(mpoles:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """
    Generates the discrete complex MT system.

    Parameters
    ----------
    mpoles : torch.Tensor
        Poles of the discrete complex MT system.
    eps : float, optional
        Accuracy of the discretization on the unit disc (default is 1e-6).

    Returns
    -------
    torch.Tensor
        The elements of the discrete complex MT system at the uniform sampling points as row vectors.
    
    Raises
    ------
    ValueError
        If the poles are not inside the unit disc.
    """
    if not isinstance(mpoles, torch.Tensor):
        raise TypeError("mpoles must be a torch.Tensor")
    if not isinstance(eps, float):
        raise TypeError("eps must be a float")

    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError("Poles must be inside the unit disc")

    m = mpoles.numel()
    t = discretize_dc(mpoles, eps)
    mts = torch.zeros(m, m+1, dtype=mpoles.dtype)

    for j in range(m):
        mts[j, :] = __mt(j, mpoles, torch.exp(1j * t))

    return mts

def __mt(n:int, mpoles:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
    """
    Compute the values of the nth Malmquist-Takenaka function at z.

    Parameters
    ----------
    n : int
        The order of the Malmquist-Takenaka function.
    mpoles : torch.Tensor
        Poles of the discrete complex MT system.
    z : torch.Tensor
        The complex points where the MT function is evaluated.

    Returns
    -------
    torch.Tensor
        The calculated values of the nth Malmquist-Takenaka function.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if not isinstance(mpoles, torch.Tensor):
        raise TypeError("mpoles must be a torch.Tensor")
    if not isinstance(z, torch.Tensor):
        raise TypeError("z must be a torch.Tensor")

    r = torch.ones_like(z)
    for k in range(1, n+1):
        r *= (z - mpoles[k-1]) / (1 - mpoles[k-1].conj() * z)
    r *= torch.sqrt(1 - torch.abs(mpoles[n])**2) / (1 - mpoles[n].conj() * z)
    return r

def mtdr_generate(length:int, mpoles:torch.Tensor, cUk:torch.Tensor, cVk:torch.Tensor) -> torch.Tensor:
    """
    Generates a function in the space spanned by the discrete real MT system.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    mpoles : torch.Tensor
        Poles of the discrete real MT system (row vector).
    cUk : torch.Tensor
        Coefficients of the linear combination to form (row vector)
        with respect to the real part of the discrete real MT system
        defined by 'mpoles'.
    cVk : torch.Tensor
        Coefficients of the linear combination to form (row vector)
        with respect to the imaginary part of the discrete real MT 
        system defined by 'mpoles'.

    Returns
    -------
    torch.Tensor
        The generated function at the uniform sampling points as a row vector.

        It is the linear combination of the discrete real MT system elements.

    Table
    -----
    +-------+-------+-------+-------+-------+-------+-------+
    |       |       | mtdr1 | mtdr1 | mtdr1 | ...   | mtdr1 |
    |       |       | mtdr2 | mtdr2 | mtdr2 | ...   | mtdr2 |
    | co1   | co2   |  SRf  |  SRf  |  SRf  | ...   |  SRf  |
    +-------+-------+-------+-------+-------+-------+-------+


    Raises
    ------
    ValueError
        If 'length' is not an integer greater than or equal to 2.

        If 'mpoles' values are greater than or equal to 1.
    """
    # Validate input parameters
    if not isinstance(length, int) or length < 2:
        raise ValueError("length must be an integer greater than or equal to 2.")
    if not isinstance(mpoles, torch.Tensor) or mpoles.dim() != 1:
        raise TypeError("mpoles must be a 1D torch.Tensor.")
    if not isinstance(cUk, torch.Tensor) or cUk.dim() != 1 or cUk.size(0) != mpoles.size(0):
        raise TypeError("cUk must be a 1D torch.Tensor with the same size as mpoles.")
    if not isinstance(cVk, torch.Tensor) or cVk.dim() != 1 or cVk.size(0) != mpoles.size(0):
        raise TypeError("cVk must be a 1D torch.Tensor with the same size as mpoles.")
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError("mpoles contains values greater than or equal to 1.")

    # Prepend 0 to mpoles as per MATLAB code
    mpoles = torch.cat((torch.tensor([0.0]), mpoles))

    # Generate the MT system elements
    mts = mt_system(length, mpoles)

    # Calculate the generated function
    SRf = cUk[0] * mts[:, 0]
    for i in range(1, mpoles.size(0)):
        SRf += 2 * cUk[i] * torch.real(mts[:, i]) + 2 * cVk[i] * torch.imag(mts[:, i])

    return SRf

from .util import discretize_dr

def mtdr_system(poles, eps=1e-6):
    """
    Generates the discrete real MT system.

    Parameters
    ----------
    poles : torch.Tensor
        Poles of the discrete real MT system.
    eps : float, optional
        Accuracy of the real discretization on the unit disc (default is 1e-6).

    Returns
    -------
    mts_re : torch.Tensor
        The real part of the discrete complex MT system at the non-equidistant discretization on the unit disc.
    mts_im : torch.Tensor
        The imaginary part of the discrete complex MT system at the non-equidistant discretization on the unit disc.

    Raises
    ------
    ValueError
        If any of the poles have a magnitude greater than or equal to 1.
    """
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Bad poles! Poles must have magnitudes less than 1.')

    mpoles = torch.cat((torch.zeros(1), poles))
    m = mpoles.size(0)
    t = discretize_dr(poles, eps)
    mts_re = torch.zeros(m, t.size(0), dtype=torch.complex64)
    mts_im = torch.zeros(m, t.size(0), dtype=torch.complex64)

    for j in range(m):
        mt_values = __mt(j - 1, mpoles, torch.exp(1j * t))
        mts_re[j, :] = mt_values.real
        mts_im[j, :] = mt_values.imag

    return mts_re, mts_im

def mt_coeffs(v: torch.Tensor, poles: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Calculate the Fourier coefficients of 'v' with respect to the 
    Malmquist-Takenaka system defined by 'poles'.

    Parameters
    ----------
    v : torch.Tensor
        An arbitrary vector.
    poles : torch.Tensor
        Poles of the rational system.

    Returns
    -------
    torch.Tensor
        The Fourier coefficients of v with respect to the Malmquist-Takenaka 
        system defined by poles.
    float
        L^2 norm of the approximation error.
    
    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(v, torch.Tensor) or v.ndim != 1:
        raise ValueError('v must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(poles, torch.Tensor) or poles.ndim != 1:
        raise ValueError('Poles must be a 1-dimensional torch.Tensor.')
    
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must be inside the unit circle!')

    # Calculate the Malmquist-Takenaka system elements
    mts = mt_system(v.size(0), poles)
    
    # Calculate coefficients and error
    co = (mts @ v.unsqueeze(1) / v.size(0)).squeeze()
    err = torch.linalg.norm(co @ mts - v).item()
    
    return co, err

def mt_generate(length: int, poles: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Generates a function in the space spanned by the MT system.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor
        Poles of the rational system (1-dimensional tensor).
    coeffs : torch.Tensor
        Coefficients of the linear combination to form (1-dimensional tensor).

    Returns
    -------
    torch.Tensor
        The generated function at the uniform sampling points as a 1-dimensional tensor.

        It is the linear combination of the MT system elements.

    Table
    -----
    +-------+-------+-------+-------+-------+-------+-------+
    |       |       |  mt1  |  mt1  |  mt1  | ...   |  mt1  |
    |       |       |  mt2  |  mt2  |  mt2  | ...   |  mt2  |
    | co1   | co2   |   v   |   v   |   v   | ...   |   v   |
    +-------+-------+-------+-------+-------+-------+-------+

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(length, int) or length < 2:
        raise ValueError('Length must be an integer greater than or equal to 2.')
    
    if not isinstance(poles, torch.Tensor) or poles.ndim != 1:
        raise ValueError('Poles must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(coeffs, torch.Tensor) or coeffs.ndim != 1:
        raise ValueError('Coeffs must be a 1-dimensional torch.Tensor.')
    
    if poles.shape[0] != coeffs.shape[0]:
        raise ValueError('Poles and coeffs must have the same number of elements.')
    
    if torch.max(torch.abs(poles)) >= 1:
        raise ValueError('Poles must be inside the unit circle!')

    # Calculate the MT system elements
    mt_sys = mt_system(length, poles)
    
    # Calculate the linear combination of the MT system elements
    v = coeffs @ mt_sys

    return v

from .util import subsample, dotdc

def mtdc_coeffs(signal: torch.Tensor, mpoles: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, float]:
    """
    Calculate the mtdc-coefficients of 'signal' with respect to the 
    discrete complex MT system given by 'mpoles'.

    Parameters
    ----------
    signal : torch.Tensor
        An arbitrary vector.
    mpoles : torch.Tensor
        Poles of the discrete complex MT system.
    eps : float
        Accuracy of the complex discretization on the unit disc.

    Returns
    -------
    torch.Tensor
        The Fourier coefficients of 'signal' with respect to the discrete 
        complex MT system defined by 'mpoles'.
    float
        L^2 norm of the approximation error.
    
    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(signal, torch.Tensor) or signal.ndim != 1:
        raise ValueError('signal must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(mpoles, torch.Tensor) or mpoles.ndim != 1:
        raise ValueError('mpoles must be a 1-dimensional torch.Tensor.')
    
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError('mpoles must be inside the unit circle!')
    
    if not isinstance(eps, float):
        raise ValueError('eps must be a float.')

    # Discretize and sample
    t = discretize_dc(mpoles, eps)
    samples = subsample(signal, t)

    # Calculate coefficients using helper functions assumed to be implemented correctly
    m = len(mpoles)
    co = torch.zeros(1, m)
    mts = mtdc_system(mpoles, eps)

    for i in range(m):
        co[0, i] = dotdc(samples, mts[i], mpoles, t)

    # Calculate error
    len_signal = len(signal)
    mts = mt_system(len_signal, mpoles)
    err = torch.norm(co @ mts - signal).item()

    return co.squeeze(), err

def mtdc_generate(length: int, mpoles: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Generates a function in the space spanned by the discrete complex MT system.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    mpoles : torch.Tensor
        Poles of the discrete complex MT system (row vector).
    coeffs : torch.Tensor
        Coefficients of the linear combination to form (row vector).

    Returns
    -------
    torch.Tensor
        The generated function at the uniform sampling points as a row vector.

        It is the linear combination of the discrete complex MT system elements.

    Table
    -----
    +-------+-------+-------+-------+-------+-------+-------+
    |       |       | mtdc1 | mtdc1 | mtdc1 | ...   | mtdc1 |
    |       |       | mtdc2 | mtdc2 | mtdc2 | ...   | mtdc2 |
    | co1   | co2   |   v   |   v   |   v   | ...   |   v   |
    +-------+-------+-------+-------+-------+-------+-------+

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(length, int) or length < 2:
        raise ValueError('length must be an integer greater than or equal to 2.')
    
    if not isinstance(mpoles, torch.Tensor) or mpoles.ndim != 1:
        raise ValueError('mpoles must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(coeffs, torch.Tensor) or coeffs.ndim != 1:
        raise ValueError('coeffs must be a 1-dimensional torch.Tensor.')
    
    if mpoles.size(0) != coeffs.size(0):
        raise ValueError('mpoles and coeffs must have the same number of elements.')
    
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError('mpoles must be inside the unit circle!')

    # Generate the MT system elements
    mts = mt_system(length, mpoles)
    v = coeffs @ mts

    return v

from .util import dotdr

def mtdr_coeffs(v: torch.Tensor, mpoles: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Calculates the mtdr-coefficients of 'v' with respect to the discrete real MT system given by 'mpoles'.

    Parameters
    ----------
    v : torch.Tensor
        An arbitrary vector.
    mpoles : torch.Tensor
        Poles of the discrete real MT system.
    eps : float, optional
        Accuracy of the real discretization on the unit disc. Default is 1e-6.

    Returns
    -------
    torch.Tensor
        The Fourier coefficients of 'v' with respect to the real part 
        of the discrete real MT system defined by 'mpoles'.
    torch.Tensor
        The Fourier coefficients of 'v' with respect to the imaginary 
        part of the discrete real MT system defined by 'mpoles'.
    float
        L^2 norm of the approximation error.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(v, torch.Tensor) or v.ndim != 1:
        raise ValueError('v must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(mpoles, torch.Tensor) or mpoles.ndim != 1:
        raise ValueError('mpoles must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(eps, float) or eps <= 0:
        raise ValueError('eps must be a positive float.')
    
    if torch.max(torch.abs(mpoles)) >= 1:
        raise ValueError('mpoles must be inside the unit circle!')

    # Calculate the mtdr system elements
    m = len(mpoles) + 1
    t = discretize_dr(mpoles, eps)
    samples = subsample(v, t)
    
    cUk = torch.zeros(m)
    cVk = torch.zeros(m)
    
    mts_re, mts_im = mtdr_system(mpoles, eps)

    for i in range(m):
        cUk[i] = dotdr(samples, mts_re[i], mpoles, t)
        cVk[i] = dotdr(samples, mts_im[i], mpoles, t)

    SRf = mtdr_generate(len(v), mpoles, cUk, cVk)
    
    err = torch.norm(SRf - v).item()

    return cUk, cVk, err
