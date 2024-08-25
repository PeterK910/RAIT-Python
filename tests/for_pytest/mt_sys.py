import torch

def mt_system(len: int, poles: torch.Tensor) -> torch.Tensor:
    """
    Generates the Malmquist-Takenaka system.

    Parameters
    ----------
    len : int
        Number of points in case of uniform sampling. Must be an integer greater than or equal to 2.
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the rational system. Must be a 1-dimensional tensor.

    Returns
    -------
    torch.Tensor
        The elements of the MT system at the uniform sampling points as row vectors.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from .util import check_poles

    # Validate input parameters
    if not isinstance(len, int):
        raise TypeError('len must be an integer.')
    
    if len < 2:
        raise ValueError('len must be an integer greater than or equal to 2.')

    check_poles(poles)

    # Calculate the MT system elements
    n=poles.size(0)
    mts = torch.zeros(n, len, dtype=torch.complex64)
    t = torch.linspace(-torch.pi, torch.pi, len + 1)[:-1]
    z = torch.exp(1j * t)

    fi = torch.ones(len)  # The product defining MT elements so far
    for j in range(n):
        co = torch.sqrt(1 - (torch.abs(poles[j]) ** 2))
        rec = 1 / (1 - torch.conj(poles[j]) * z)
        lin = co * rec
        bla = (z - poles[j]) * rec
        mts[j, :] = lin * fi
        fi = fi * bla

    return mts



def mtdc_system(mpoles:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """
    Generates the discrete complex MT system.

    Parameters
    ----------
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the discrete complex MT system. Must be a 1-dimensional tensor.
    eps : float, optional
        Accuracy of the discretization on the unit disc (default is 1e-6).

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The elements of the discrete complex MT system at the uniform sampling points as row vectors.
    
    Raises
    ------
    ValueError
        If the poles are not inside the unit disc.
    """
    from .util import check_poles ,discretize_dc
    # Validate input parameters
    check_poles(mpoles)
    if not isinstance(eps, float):
        raise TypeError("eps must be a float")
    if eps <= 0:
        raise ValueError("eps must be positive")

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
    for k in range(n):
        r *= (z - mpoles[k]) / (1 - torch.conj(mpoles[k]) * z)
    r *= torch.sqrt(1 - torch.abs(mpoles[n])**2) / (1 - torch.conj(mpoles[n]) * z)
    return r

def mtdr_generate(length:int, mpoles:torch.Tensor, cUk:torch.Tensor, cVk:torch.Tensor) -> torch.Tensor:
    """
    Generates a function in the space spanned by the discrete real MT system.

    NOTE: Prints a warning if the imaginary part of the generated function is non-zero.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.

    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the discrete real MT system (1-dimensional tensor). Must be inside the unit circle.
        
        Must have 1 less element than 'cUk' and 'cVk' separately.

    cUk : torch.Tensor, dtype=torch.float64
        Coefficients of the linear combination to form (1-dimensional tensor) with respect to the real part of the discrete real MT system defined by 'mpoles'.

        Must have the same size as 'cVk1 and 1 more element than 'mpoles'.

    cVk : torch.Tensor, dtype=torch.float64
        Coefficients of the linear combination to form (1-dimensional tensor) with respect to the imaginary part of the discrete real MT system defined by 'mpoles'.

        Must have the same size as 'cUk' and 1 more element than 'mpoles'.

    Returns
    -------
    torch.Tensor, dtype=torch.float32
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
    from .util import check_poles
    # Validate input parameters
    if not isinstance(length, int):
        raise TypeError("length must be an integer.")
    if length < 2:
        raise ValueError("length must be an integer greater than or equal to 2.")
    
    check_poles(mpoles)

    if not isinstance(cUk, torch.Tensor):
        raise TypeError("cUk must be a torch.Tensor.")
    if cUk.ndim != 1:
        raise ValueError("cUk must be a 1-dimensional torch.Tensor.")
    if not cUk.is_floating_point():
        raise TypeError("cUk must be a float tensor.")
    
    if not isinstance(cVk, torch.Tensor):
        raise TypeError("cVk must be a torch.Tensor.")
    if cVk.ndim != 1:
        raise ValueError("cVk must be a 1-dimensional torch.Tensor.")
    if not cVk.is_floating_point():
        raise TypeError("cVk must be a float tensor.")
    # Check the size of mpoles, cUk, and cVk
    if mpoles.size(0) != cUk.size(0) - 1:
        raise ValueError("mpoles must have 1 less element than cUk.")
    if mpoles.size(0) != cVk.size(0) - 1:
        raise ValueError("mpoles must have 1 less element than cVk.")
    #if these two above conditions are satisfied, then cUk and cVk have the same size

    # Prepend 0 to mpoles
    mpoles = torch.cat((torch.tensor([0.0]), mpoles))

    # Generate the MT system elements
    mts = mt_system(length, mpoles)
    # Calculate the generated function
    # since mpoles[0] = 0, the resulting mts[0] is just 1-s. TODO: verify if this is correct mathematically
    # that and the fact that cUk is float64, the type of SRf is guaranteed to be float64
    SRf = cUk[0] * mts[0]
    for i in range(1, mpoles.size(0)):
        #even these operations are guaranteed to keep the type of SRf float64
        SRf += 2 * cUk[i] * torch.real(mts[i]) + 2 * cVk[i] * torch.imag(mts[i])
    #convert SRf to float64 for type consistency
    
    #in case the above operations did not guarantee the type of SRf to be float64
    if SRf.imag.max() > 0:
        print(f"Warning: SRf has non-zero imaginary part. SRf.imag: {SRf.imag}")
    SRf = torch.real(SRf)
    return SRf



def mtdr_system(poles: torch.Tensor, eps:float=1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the discrete real MT system.

    Parameters
    ----------
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the discrete real MT system. Must be a 1-dimensional tensor.
    eps : float, optional
        Accuracy of the real discretization on the unit disc (default is 1e-6).

    Returns
    -------
    mts_re : torch.Tensor, dtype=torch.float64
        The real part of the discrete complex MT system at the non-equidistant discretization on the unit disc.
    mts_im : torch.Tensor, dtype=torch.float64
        The imaginary part of the discrete complex MT system at the non-equidistant discretization on the unit disc.

    Raises
    ------
    ValueError
        If any of the poles have a magnitude greater than or equal to 1.
    """
    from .util import check_poles ,discretize_dr
    # Validate input parameters
    check_poles(poles)
    if not isinstance(eps, float):
        raise TypeError("eps must be a float")
    if eps <= 0:
        raise ValueError("eps must be positive")

    mpoles = torch.cat((torch.zeros(1), poles))
    m = mpoles.size(0)
    t = discretize_dr(poles, eps)
    mts_re = torch.zeros(m, t.size(0), dtype=torch.float64)
    mts_im = torch.zeros(m, t.size(0), dtype=torch.float64)
    for j in range(m):
        mt_values = __mt(j, mpoles, torch.exp(1j * t))
        mts_re[j, :] = mt_values.real
        mts_im[j, :] = mt_values.imag

    return mts_re, mts_im

def mt_coeffs(v: torch.Tensor, poles: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Calculate the Fourier coefficients of 'v' with respect to the 
    Malmquist-Takenaka system defined by 'poles'.

    Parameters
    ----------
    v : torch.Tensor, dtype=torch.complex64
        An arbitrary vector. Must be a 1-dimensional tensor.
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the rational system. Must be a 1-dimensional tensor. Elements must be inside the unit circle.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The Fourier coefficients of v with respect to the Malmquist-Takenaka 
        system defined by poles.
    float
        L^2 norm of the approximation error.
    
    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from .util import check_poles, conj_trans
    # Validate input parameters
    if not isinstance(v, torch.Tensor):
        raise TypeError('v must be a torch.Tensor.')
    if v.ndim != 1:
        raise ValueError('v must be a 1-dimensional torch.Tensor.')
    if not v.is_complex():
        raise TypeError('v must be a complex tensor.')
    
    check_poles(poles)

    # Calculate the Malmquist-Takenaka system elements
    mts = mt_system(v.size(0), poles)
    
    # Calculate coefficients and error
    co = conj_trans(torch.matmul(mts, conj_trans(v)) / v.size(0))
    err = torch.linalg.norm(co @ mts - v).item()
    
    return co, err

def mt_generate(length: int, poles: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Generates a function in the space spanned by the MT system.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the rational system (1-dimensional tensor). Must be inside the unit circle.
    coeffs : torch.Tensor, dtype=torch.complex64
        Coefficients of the linear combination to form (1-dimensional tensor).

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
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

    from .util import check_poles
    
    # Validate input parameters
    if not isinstance(length, int):
        raise TypeError('length must be an integer.')
    if length < 2:
        raise ValueError('length must be greater than or equal to 2.')
    
    check_poles(poles)
    
    if not isinstance(coeffs, torch.Tensor):
        raise TypeError('coeffs must be a torch.Tensor.')
    if coeffs.ndim != 1:
        raise ValueError('coeffs must be a 1-dimensional torch.Tensor.')
    if not coeffs.is_complex():
        raise TypeError('coeffs must be a complex tensor.')
    
    #coeffs must have the same number of elements as poles
    if poles.size(0) != coeffs.size(0):
        raise ValueError('poles and coeffs must have the same number of elements.')

    # Calculate the MT system elements
    mt_sys = mt_system(length, poles)
    
    # Calculate the linear combination of the MT system elements
    v = coeffs @ mt_sys

    return v



def mtdc_coeffs(signal: torch.Tensor, mpoles: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, float]:
    """
    Calculate the mtdc-coefficients of 'signal' with respect to the 
    discrete complex MT system given by 'mpoles'.

    Parameters
    ----------
    signal : torch.Tensor, dtype=torch.complex64
        An arbitrary 1-dimensional tensor.
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the discrete complex MT system. 1-dimensional tensor. Must be inside the unit circle.
    eps : float, optional
        Accuracy of the complex discretization on the unit disc.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The Fourier coefficients of 'signal' with respect to the discrete 
        complex MT system defined by 'mpoles'.
    float
        L^2 norm of the approximation error.
    
    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from .util import subsample, dotdc, discretize_dc, check_poles

    # Validate input parameters
    if not isinstance(signal, torch.Tensor):
        raise TypeError('signal must be a torch.Tensor.')
    if signal.ndim != 1:
        raise ValueError('signal must be a 1-dimensional torch.Tensor.')
    if not signal.is_complex():
        raise TypeError('signal must be a complex tensor.')
    
    check_poles(mpoles)
    
    if not isinstance(eps, float):
        raise TypeError('eps must be a float.')
    if eps <= 0:
        raise ValueError('eps must be a positive float.')

    # Discretize and sample
    t = discretize_dc(mpoles, eps)
    samples = subsample(signal, t)

    # Calculate coefficients using helper functions assumed to be implemented correctly
    m = len(mpoles)
    co = torch.zeros(m, dtype=torch.complex64)
    mts = mtdc_system(mpoles, eps)

    for i in range(m):
        co[i] = dotdc(samples, mts[i], mpoles, t)

    # Calculate error
    len_signal = len(signal)
    mts = mt_system(len_signal, mpoles)
    err = torch.linalg.norm(co @ mts - signal).item()

    return co, err

def mtdc_generate(length: int, mpoles: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Generates a function in the space spanned by the discrete complex MT system.

    TODO: The function uses mt_system() instead of mtdc_system() to generate the MT system elements. Verify if this is intended.
    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the discrete complex MT system (1-dimensional tensor). Must be inside the unit circle.
    coeffs : torch.Tensor, dtype=torch.complex64
        Coefficients of the linear combination to form (1-dimensional tensor).

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The generated function at the uniform sampling points as a 1-dimensional tensor.

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

    from .util import check_poles
    
    # Validate input parameters
    if not isinstance(length, int):
        raise TypeError('length must be an integer.')
    if length < 2:
        raise ValueError('length must be greater than or equal to 2.')
    
    check_poles(mpoles)
    
    if not isinstance(coeffs, torch.Tensor):
        raise TypeError('coeffs must be a torch.Tensor.')
    if coeffs.ndim != 1:
        raise ValueError('coeffs must be a 1-dimensional torch.Tensor.')
    if not coeffs.is_complex():
        raise TypeError('coeffs must be a complex tensor.')
    
    #coeffs must have the same number of elements as mpoles
    if mpoles.size(0) != coeffs.size(0):
        raise ValueError('mpoles and coeffs must have the same number of elements.')

    # Generate the MT system elements
    mts = mt_system(length, mpoles)
    v = coeffs @ mts

    return v



def mtdr_coeffs(v: torch.Tensor, mpoles: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Calculates the mtdr-coefficients of 'v' with respect to the discrete real MT system given by 'mpoles'.

    TODO: verify if v should be complex or real. Right now, it is real as per test.m from matlab code.

    TODO: verify if output cUk and cVk should be complex or real. Right now, they are real as per test.m from matlab code.
    
    Parameters
    ----------
    v : torch.Tensor, dtype=torch.float64
        An arbitrary 1-dimensional tensor.
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the discrete real MT system. 1-dimensional tensor. Must be inside the unit circle.
    eps : float, optional
        Accuracy of the real discretization on the unit disc. Default is 1e-6.

    Returns
    -------
    torch.Tensor, dtype=torch.float64
        The Fourier coefficients of 'v' with respect to the real part 
        of the discrete real MT system defined by 'mpoles'.
    torch.Tensor, dtype=torch.float64
        The Fourier coefficients of 'v' with respect to the imaginary 
        part of the discrete real MT system defined by 'mpoles'.
    float
        L^2 norm of the approximation error.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from .util import discretize_dr, subsample, dotdr, check_poles

    # Validate input parameters
    if not isinstance(v, torch.Tensor):
        raise TypeError('v must be a torch.Tensor.')
    if v.ndim != 1:
        raise ValueError('v must be a 1-dimensional torch.Tensor.')
    if not v.is_floating_point():
        raise TypeError('v must be a float tensor.')
    
    check_poles(mpoles)
    
    if not isinstance(eps, float):
        raise TypeError('eps must be a float.')
    if eps <= 0:
        raise ValueError('eps must be a positive float.')

    # Calculate the mtdr system elements
    m = len(mpoles) + 1
    t = discretize_dr(mpoles, eps)
    #convert v to complex64 for subsample function
    v = v.to(torch.complex64)
    samples = subsample(v, t)
    
    cUk = torch.zeros(m, dtype=torch.float64)
    cVk = torch.zeros(m, dtype=torch.float64)
    
    mts_re, mts_im = mtdr_system(mpoles, eps)
    mts_re = mts_re.to(torch.complex64)
    mts_im = mts_im.to(torch.complex64)

    for i in range(m):
        #warning is thrown here because of the complex64 type of mts_re[i] and samples
        #whereas cUk is float64. See todo above.
        cUk[i] = dotdr(samples, mts_re[i], mpoles, t)
        cVk[i] = dotdr(samples, mts_im[i], mpoles, t)

    SRf = mtdr_generate(len(v), mpoles, cUk, cVk)
    err = torch.linalg.norm(SRf - v).item()

    return cUk, cVk, err
