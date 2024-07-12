import numpy as np

# paradoxi: lamb teleutaio
def light(point: np.ndarray, normal: np.ndarray, vcolor: np.ndarray, cam_pos: np.ndarray, ka: float, kd: float, ks: float, n: int, lpos: np.ndarray, lint: np.ndarray) -> np.ndarray:
    # ambient light calculation
    lamb = lint[:,-1]
    
    total_light = np.zeros((3,3))
    total_light[0] = ka * lamb

    # iterate over every light source
    view_dir = (cam_pos - point) / np.linalg.norm(cam_pos - point)
    for i in range(lpos.shape[1]):
        light_dir = lpos[:,i].reshape((3,1)) - point
        light_dir = light_dir / np.linalg.norm(light_dir)
 
        total_light[1] += kd * np.dot(normal.T[0], light_dir.T[0]) * lint[:,i]

        reflect_dir = 2 * np.dot(normal.T[0], light_dir.T[0]) * normal - light_dir
        reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)
        total_light[2] += ks * (np.dot(view_dir.T[0], reflect_dir.T[0]) ** n) * lint[:,i]
        total_light = np.clip(total_light, 0, 1)

    return np.clip(np.sum(total_light, axis=0) * vcolor, 0, 1) 


