import torch

def mlf_system(length: int, mpoles: torch.Tensor) -> torch.Tensor:
    """
    Generates the modified basic rational function system defined by mpoles.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling. Must be greater than or equal to 2.
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the modified basic rational system. Must be a 1D tensor.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The elements of the modified basic rational function system
        at the uniform sampling points as row vectors.

    Raises
    ------
    ValueError
        If the number of poles is not 1 or the length is less than 2.
        Also, if the poles are not inside the unit circle.
    """
    from .util import check_poles, multiplicity
    # Validate input parameters
    if not isinstance(length, int):
        raise TypeError('Length must be an integer.')
    if length < 2:
        raise ValueError('Length must be greater than or equal to 2.')
    check_poles(mpoles)

    mlf = torch.zeros(mpoles.numel(), length, dtype=torch.complex64)
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
        Number of points in case of uniform sampling. Must be greater than or equal to 2.
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the rational system. Must be a 1D tensor.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The elements of the linearly independent system at the uniform
        sampling points as row vectors.

    Raises
    ------
    ValueError
        If the number of poles is not 1 or the length is less than 2.
        Also, if the poles are not inside the unit circle.
    """
    from .util import check_poles
    # Validate input parameters
    if not isinstance(length, int):
        raise TypeError('Length must be an integer.')
    if length < 2:
        raise ValueError('Length must be greater than or equal to 2.')
    check_poles(poles)

    lfs = torch.zeros(poles.numel(), length, dtype=torch.complex64) # complex dtype
    t = torch.linspace(-torch.pi, torch.pi, length + 1)[:-1]
    z = torch.exp(1j * t)
    for j in range(poles.numel()):
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
    for k in range(n+1):
        if torch.allclose(v[n], v[k]):
            m += 1
    return m

def mlfdc_system(mpoles:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """
    Generates the discrete modified basic rational system.

    Parameters
    ----------
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the discrete modified basic rational system. Must be a 1D tensor.
    eps : float, optional
        Accuracy of the discretization on the unit disc (default is 1e-6).

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The elements of the discrete modified basic rational system at the
        uniform sampling points as row vectors.

    Raises
    ------
    ValueError
        If the poles are not inside the unit disc.
    """
    from .util import check_poles,multiplicity, discretize_dc
    # Validate input parameters
    check_poles(mpoles)

    if not isinstance(eps, float):
        raise TypeError('eps must be a float.')
    if eps <= 0:
        raise ValueError('eps must be positive.')

    m = mpoles.numel()
    mlf = torch.zeros(m, m+1, dtype=mpoles.dtype)
    t = discretize_dc(mpoles, eps)

    spoles, multi = multiplicity(mpoles)

    for j in range(len(multi)):
        for k in range(multi[j]):
            col = sum(multi[:j]) + k
            #compared to matlab version, since k is 0-based here, any k used in exponentiation is increased by 1 here
            mlf[col, :] = torch.exp(1j * t) ** k / (1 - spoles[j].conj() * torch.exp(1j * t)) ** (k+1)

    return mlf

def mlf_generate(length:int, poles:torch.Tensor, coeffs:torch.Tensor) -> torch.Tensor:
    """
    Compute the values of the poisson function at (r,t).

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the modified basic rational system (1-dimensional tensor). Must be inside the unit circle.

        Must have the same number of elements as 'coeffs'.
    coeffs : torch.Tensor, dtype=torch.complex64
        Coefficients of the linear combination to form (1-dimensional tensor).

        Must have the same number of elements as 'poles'.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The generated function at the uniform sampling points as a 1D tensor.

        It is the linear combination of the discrete real MT system elements.

    Table
    -----
    +-------+-------+-------+-------+-------+-------+-------+
    |       |       | mlf1  | mlf1  | mlf1  | ...   | mlf1  |
    |       |       | mlf2  | mlf2  | mlf2  | ...   | mlf2  |
    | co1   | co2   |   v   |   v   |   v   | ...   |   v   |
    +-------+-------+-------+-------+-------+-------+-------+

    Raises
    ------
    ValueError
        If input parameters are incorrect or if poles are outside the valid range.
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
    v : torch.Tensor, dtype=torch.complex64
        An arbitrary 1D tensor.
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the modified basic rational system. Must be a 1D tensor. Each pole must be inside the unit circle.

    Returns
    -------
    co : torch.Tensor, dtype=torch.complex64
        The Fourier coefficients of v with respect to the modified basic rational system defined by 'poles'.
    err : float
        L^2 norm of the approximation error.

    Raises
    ------
    ValueError
        If input parameters are incorrect or if poles are outside the valid range.
    """

    from .biort_sys import biort_system
    from .util import check_poles, conj_trans
    
    # Validate input parameters
    if not isinstance(v, torch.Tensor):
        raise TypeError('v must be a torch.Tensor.')
    if v.dim() != 1:
        raise ValueError('v must be a 1D tensor.')
    if not v.is_complex():
        raise TypeError('v must be a complex tensor.')
    
    check_poles(poles)

    # Calculate biorthogonal system elements 
    bts = biort_system(v.size(0), poles)
    
    # Calculate Fourier coefficients
    co = conj_trans(torch.matmul(bts, conj_trans(v)) / v.size(0))
    # Calculate modified rational system elements
    mlf = mlf_system(v.size(0), poles)
    # Calculate approximation error
    err = torch.linalg.norm(torch.matmul(co.t(), mlf) - v)


    return co, err.item()

def lf_generate(length: int, poles: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Generate a function in the space spanned by the LF system.

    Parameters
    ----------
    length : int
        Number of points in case of uniform sampling.
    poles : torch.Tensor, dtype=torch.complex64
        Poles of the modified basic rational system (1-dimensional tensor). Must be inside the unit circle.

        Must have the same number of elements as 'coeffs'.
    coeffs : torch.Tensor, dtype=torch.complex64
        Coefficients of the linear combination to form (1-dimensional tensor).

        Must have the same number of elements as 'poles'.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The generated function at the uniform sampling points as a row vector.
    
        It is the linear combination of the LF system elements.

    Table
    -----
    +-------+-------+-------+-------+-------+-------+-------+
    |       |       |  lf1  |  lf1  |  lf1  | ...   |  lf1  |
    |       |       |  lf2  |  lf2  |  lf2  | ...   |  lf2  |
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

    
    # Generate the LF system elements
    lf_sys = lf_system(length, poles)
    
    # Perform matrix multiplication to generate the function
    v = coeffs @ lf_sys
    
    return v

def mlfdc_coeffs(signal: torch.Tensor, mpoles: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, float]:
    """
    Calculates the mlfdc-coefficients of 'signal' with respect to the
    discrete modified basic rational system given by 'mpoles'.

    Parameters
    ----------
    signal : torch.Tensor, dtype=torch.complex64
        An arbitrary 1-dimensional tensor.
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the modified basic rational system (1-dimensional tensor). Each pole must be inside the unit circle.
    eps : float, optional
        Accuracy of the discretization on the unit disc (default is 1e-6).

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The Fourier coefficients of 'signal' with respect to the discrete 
        modified basic rational system defined by 'mpoles'.
    float
        L^2 norm of the approximation error.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    from .blaschke import arg_inv
    from .util import subsample, dotdc, check_poles
    from .biort_sys import biortdc_system

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

    # Subsample signal and calculate coefficients
    m = mpoles.numel()
    #substracting eps/1000 to avoid pi in linspace for arg_inv
    z = torch.linspace(-torch.pi, torch.pi - eps/1000, m + 1, dtype=torch.float64)
    t = arg_inv(mpoles, z, eps)
    samples = subsample(signal, t)
    
    co = torch.zeros(m, dtype=torch.complex64)
    bts = biortdc_system(mpoles, eps)

    for i in range(m):
        co[i] = dotdc(samples, bts[i], mpoles, t)
    # Calculate error
    len_signal = signal.numel()
    mlf = mlf_system(len_signal, mpoles)
    
    err = torch.linalg.norm(torch.matmul(co, mlf) - signal).item()

    return co, err

def mlfdc_generate(mpoles: torch.Tensor, coeffs: torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    """
    Generates a function in the space spanned by the discrete modified basic rational function system.

    Parameters
    ----------
    mpoles : torch.Tensor, dtype=torch.complex64
        Poles of the discrete modified basic rational system (1-dimensional tensor). Must be inside the unit circle.
    coeffs : torch.Tensor, dtype=torch.complex64
        Coefficients of the linear combination to form (1-dimensional tensor).
    eps : float, optional
        Accuracy of the discretization on the unit disc. Default is 1e-6.

    Returns
    -------
    torch.Tensor, dtype=torch.complex64
        The generated function at the uniform sampling points as a 1-dimensional tensor.

        It is the linear combination of the discrete modified basic rational system elements.

    Table
    -----
    +-------+-------+-------+-------+-------+-------+-------+
    |       |       | mlfdc1| mlfdc1| mlfdc1| ...   | mlfdc1|
    |       |       | mlfdc2| mlfdc2| mlfdc2| ...   | mlfdc2|
    | co1   | co2   |   v   |   v   |   v   | ...   |   v   |
    +-------+-------+-------+-------+-------+-------+-------+

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    from .util import check_poles
    
    # Validate input parameters
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
    
    if not isinstance(eps, float):
        raise TypeError('eps must be a float.')
    if eps <= 0:
        raise ValueError('eps must be a positive float.')

    # Generate the function
    mlf = mlfdc_system(mpoles,eps)
    
    v = coeffs @ mlf

    return v
