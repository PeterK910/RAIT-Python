import torch

def coords2params(k: torch.Tensor) -> torch.Tensor:
    """
    Maps coordinates in R^2 to parameters in C. One row.

    Parameters
    ----------
    k : torch.Tensor
        Row vector of coordinate pairs in R^2.

    Returns
    -------
    torch.Tensor
        Row vector of corresponding parameters in C.

    Raises
    ------
    ValueError
        If the input is not a 1D row vector with an even number of elements.
    """
    if not isinstance(k, torch.Tensor):
        raise TypeError("Input k must be a torch.Tensor")
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
