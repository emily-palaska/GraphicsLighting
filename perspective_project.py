import numpy as np

def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
# Project the specified 3d points pts on the image plane, according to a pinhole perspective projection model.
#  Input:
#    pts: Nx3 np array of the 3d points in WCS
#    focal: float number of camera lense distance ratio
#    R: 3x3 np array representing the rotation matrix
#    t: 1x3 np vector of the camera translation vector
#  Output:
#    pts_proj: Nx2 np array of the 2d projected points in the camera plane

    # Handle dimensions
    if pts.shape[0] != 3:
        pts = pts.T
    if t.shape[0] != 3:
        t = t.T

    # Transform points to camera coordinates
    pts_cam = R.T @ (pts.T - t.T).T
    # Handle exception of points being too close (z = 0) to the plane
    if 0 in pts_cam[2]:
        print("Invalid arguments")
        return None
    
    # Extract depth
    depth = pts_cam[-1]
    
    # Project them to the image plane and return the results
    pts_proj = focal * pts_cam[:2, :] / depth
    return (pts_proj, depth.astype(int))



    