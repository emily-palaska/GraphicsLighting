import numpy as np
import matplotlib.pyplot as plt
from shading import render_object

# Load file
data = np.load('h3.npy', allow_pickle=True).item()

# Export keys in seperate variables
verts = data['verts']
vertex_colors = data['vertex_colors']
face_indices = data['face_indices']
uvs = data['uvs']
face_uv_indices = data['face_uv_indices']
cam_eye = data['cam_eye']
cam_up = data['cam_up']
cam_lookat = data['cam_lookat']
ka = data['ka']
kd = data['kd']
ks = data['ks']
n = data['n']
light_positions = data['light_positions']
light_intensities = data['light_intensities']
Ia = data['Ia']
M = data['M']
N = data['N']
W = data['W']
H = data['H']
bg_color = data['bg_color']
focal = data['focal']

# Dimension handling so that all vectors are 3xN
cam_eye = np.array([cam_eye]).T
cam_up = np.array([cam_up]).T
cam_lookat = np.array([cam_lookat]).T
light_intensities = np.array(light_intensities).T
light_positions = np.array(light_positions)
Ia = np.array([Ia]).T
bg_color = np.array(bg_color)
shader = 'phong'

# Redner object via implemented function
img = render_object(shader, focal, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia)

# Results visualization
plt.imshow(img)
plt.show()
plt.imsave(f'result_{shader}.png', img)
