# Function that performs linear interpolation between two vectors
# Warninig: It is assumed that the target point p belongs in the given line segment (p1p2)
#
# Input:
#    p1: coordinates of vector V1
#    p2: coordinates of vector V2
#    V1: vector value of V1
#    V2: vector value of V2
#    coord: the target coordinate of p (based on dim)
#    dim: defines which coordinate of p is given (x->1, y->2)
#
# Output:
#    V: the result of the interpolation (vector value) at point p
def vector_interp(p1, p2, V1, V2, coord, dim):
    # Handle exception: vertical or horizontal line with invalid arguments
    if (p1[0] == p2[0] and dim == 1) or (p1[1] == p2[1] and dim == 2) :
        return None

    # Calculate the interpolation parameter t
    if dim == 1:
        t = (coord - p1[0]) / (p2[0] - p1[0])
    elif dim == 2:
        t = (coord - p1[1]) / (p2[1] - p1[1])
    else:
        raise ValueError("Invalid dimension. dim should be 1 or 2.")
    
    # Interpolate the value
    return (1 - t) * V1 + t * V2