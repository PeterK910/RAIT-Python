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
    if not k.is_floating_point():
        raise TypeError("Input k must be a float tensor")
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
    if not k.is_floating_point():
        raise TypeError('k must be a float tensor.')
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
    from .util import check_poles
    # Validate input parameters
    check_poles(p)
    if not isinstance(m, torch.Tensor):
        raise TypeError('m must be a torch.Tensor.')
    if m.dtype != torch.int64:
        raise TypeError('m must be a tensor of dtype int64.')
    if m.ndim != 1:
        raise ValueError('m must be a 1D tensor.')
    if m.min() < 0:
        raise ValueError('m must contain only non-negative elements.')
    if p.size(0) != m.size(0):
        raise ValueError('Length of p and m must be equal.')
    
    # Check if p contains unique poles
    unique_poles = torch.tensor([p[0]], dtype=p.dtype)
    for i in range(1,p.numel()):
        is_unique = True
        for j in range(unique_poles.numel()):
            if torch.allclose(p[i], unique_poles[j]):
                is_unique = False
                break
        if is_unique:
            unique_poles = torch.cat((unique_poles, p[i].unsqueeze(0)))
            #print(f"unique poles gained a pole, current value: {unique_poles}")
    if unique_poles.numel() != p.numel():
        raise ValueError('Poles in p must be unique.')

    n = p.size(0)
    pp = torch.zeros((1, int(m.sum())), dtype=p.dtype)

    innerIndex = 0
    for i in range(n):
        pp[0, innerIndex:innerIndex+m[i]] = p[i] * torch.ones((1, int(m[i])), dtype=p.dtype)
        innerIndex += m[i]

    return pp.squeeze()

def periodize_poles(p: torch.Tensor, m: int) -> torch.Tensor:
    """
    NOTE: This function is unnecessary in python, as torch.repeat already exists. It is kept for compatibility with the original code.

    Duplicates periodically the elements of 'p' 'm' times.

    Parameters
    ----------
    p : torch.Tensor
        A 1-dimensional tensor that contains the poles.
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
    
    if not isinstance(m, int) or m < 0:
        raise ValueError('m must be a non-negative integer.')

    # Duplicate the poles 'm' times
    pp = p.repeat(m)

    return pp
