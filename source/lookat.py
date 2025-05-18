import numpy as np

def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
# Calculate the camera's view matrix ,i.e. its coordinate frame transformation
#  Input:
#    eye: 1x3 np vector specifying the point of the camera
#    up: 1x3 np vector for the up vector
#    target: 1x3 np vector for the 
#  Output:
#    R: 3x3 np rotation matrix
#    t: 1x3 np translation vector

    # Handle every combination of vector dimensions
    if eye.shape == (3,):
        eye = np.array([eye]).T
    elif eye.shape == (1,3):
        eye = eye.T

    if up.shape == (3,):
        up = np.array([up]).T
    elif up.shape == (1,3):
        up = up.T

    if target.shape == (3,):
        target = np.array([target]).T
    elif target.shape == (1,3):
        target = target.T

    # Calculate the three normalized camera vectors
    zc = target.T - eye.T
    zc = zc / np.linalg.norm(zc)
    yc = up.T - np.dot(up.T[0], zc[0]) * zc
    yc = yc / np.linalg.norm(yc)
    xc = np.cross(yc[0], zc[0])
    xc = xc / np.linalg.norm(xc)
    # Add them to the rotation matrix and return the resutls
    return (np.array([xc, yc[0], zc[0]]).T, eye)