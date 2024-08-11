import torch

def coords2params(k: torch.Tensor) -> torch.Tensor:
    """
    Maps coordinates in R^2 to parameters in C. One row.

    Parameters
    ----------
    k : torch.Tensor, dtype=float64
        1D tensor of coordinate pairs in R^2. Must have an even number of elements.

    Returns
    -------
    torch.Tensor, dtype=complex64
        1D tensor of corresponding parameters in C.

    Raises
    ------
    ValueError
        If the input is not a 1D row vector with an even number of elements.
    """
    if not isinstance(k, torch.Tensor):
        raise TypeError("Input k must be a torch.Tensor")
    if k.dtype != torch.float64:
        raise TypeError("Input k must be a tensor of dtype float64")
    if k.dim() != 1 or k.size(0) % 2 != 0:
        raise ValueError("Input k must be a 1D row vector with an even number of elements")

    parnum = k.size(0) // 2
    p = torch.zeros(parnum, dtype=torch.complex64)

    for j in range(parnum):
        u = k[2*j]
        v = k[2*j + 1]
        r = torch.sqrt(u**2 + v**2)
        x = u / torch.sqrt(1 + r**2)
        y = v / torch.sqrt(1 + r**2)
        z = torch.complex(x, y)
        p[j] = z

    return p

def coords2params_all(k: torch.Tensor) -> torch.Tensor:
    """
    Maps coordinates in R^2 to parameters in ID.
    TODO: should every row contain the same number of elements?
    Parameters
    ----------
    k : torch.Tensor, dtype=float64
        Matrix of coordinate pairs in R^2, rows are considered as
        vertices of the simplex.
        Must be a 2D tensor with an even number of columns.

    Returns
    -------
    torch.Tensor, dtype=complex64
        2D tensor of corresponding parameters in ID.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(k, torch.Tensor):
        raise TypeError('k must be a torch.Tensor.')
    if k.dtype != torch.float64:
        raise TypeError('k must be a tensor of dtype float64.')
    if k.ndim != 2 or k.size(1) % 2 != 0:
        raise ValueError('k must be a 2D tensor with an even number of columns.')

    # Initialize output tensor
    vertnum = k.size(0)
    parnum = k.size(1) // 2
    p = torch.zeros(vertnum, parnum, dtype=torch.complex64)

    # Map coordinates to parameters
    for i in range(vertnum):
        p[i, :] = coords2params(k[i, :])

    return p

def multiply_poles(p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Duplicates the elements of 'p' by the elements of 'm'.

    Parameters
    ----------
    p : torch.Tensor
        Row vector that contains a pole only once.
    m : torch.Tensor
        Multiplicities related to the pole vector 'p'.

    Returns
    -------
    torch.Tensor
        Vector of the poles that contains the ith element of 'p' 
        only 'm(i)' times.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(p, torch.Tensor) or p.ndim != 1:
        raise ValueError('p must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(m, torch.Tensor) or m.ndim != 1 or m.dtype != torch.int32:
        raise ValueError('m must be a 1-dimensional torch.Tensor with integer elements.')
    
    if p.size(0) != m.size(0):
        raise ValueError('Length of p and m must be equal.')

    n = p.size(0)
    pp = torch.zeros((1, int(m.sum())), dtype=p.dtype)

    innerIndex = 0
    for i in range(n):
        pp[0, innerIndex:innerIndex+m[i]] = p[i] * torch.ones((1, int(m[i])), dtype=p.dtype)
        innerIndex += m[i]

    return pp.squeeze()

def periodize_poles(p: torch.Tensor, m: int) -> torch.Tensor:
    """
    Duplicates periodically the elements of 'p' 'm' times.

    Parameters
    ----------
    p : torch.Tensor
        A row vector that contains the poles.
    m : int
        Integer factor of duplication.

    Returns
    -------
    torch.Tensor
        Vector of the poles that contains 'p' sequentially.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # Validate input parameters
    if not isinstance(p, torch.Tensor) or p.ndim != 1:
        raise ValueError('p must be a 1-dimensional torch.Tensor.')
    
    if not isinstance(m, int) or m < 1:
        raise ValueError('m must be a positive integer.')

    # Duplicate the poles 'm' times
    pp = p.repeat(m)

    return pp
