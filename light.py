import numpy as np

def light(point: np.ndarray, normal: np.ndarray, vcolor: np.ndarray, cam_pos: np.ndarray, ka: float, kd: float, ks: float, n: int, lpos: np.ndarray, lint: np.ndarray) -> np.ndarray:
    """
    Uses the full lighting model for a specific point
    
    Agrs:
        - point: 3d coordinates of the point to be lighted
        - normal: normal vector at the specific point
        - vcolor: color of the point
        - cam_pos: vector of the 3d camera coordinates
        - ka: ambient lighting constant
        - kd: diffuse lighting constant
        - ks: specular lighting constant
        - n: phong constant
        - lpos: light source positions
        - lint: light source intensities

    Returns:
        - The calculated light as a numpy vector
    """
    
    # Initialization
    lamb = lint[:,-1]
    diffuse_light = np.zeros((lpos.shape[1], 3))
    specualr_light = np.zeros((lpos.shape[1], 3))
    
    ambient_light = np.clip(ka * lamb * vcolor, 0, 1)
    
    # Light source iteration
    view_dir = (cam_pos - point) / np.linalg.norm(cam_pos - point)

    for i in range(lpos.shape[1]):
        light_dir = lpos[:,i].reshape((3,1)) - point
        light_dir = light_dir / np.linalg.norm(light_dir)
 
        diffuse_light[i] = kd * np.dot(normal.T[0], light_dir.T[0]) * lint[:,i] * vcolor

        reflect_dir = 2 * np.dot(normal.T[0], light_dir.T[0]) * normal - light_dir
        reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)

        specualr_light[i] = ks * (np.dot(view_dir.T[0], reflect_dir.T[0]) ** n) * lint[:,i] * vcolor
    diffuse_light = np.clip(diffuse_light, 0, 1)
    specualr_light = np.clip(specualr_light, 0, 1)

    total_light = ambient_light + np.sum(diffuse_light, axis=0) + np.sum(specualr_light, axis=0)
    return np.clip(total_light, 0, 1) 


