import numpy as np
from light import light
from perspective_project import perspective_project
from lookat import lookat
from g_shading import *
from rasterize import rasterize
import matplotlib.pyplot as plt

def calculate_normals(verts:np.ndarray, faces:np.ndarray) -> np.ndarray:
    vertsn = np.zeros(verts.shape)
    for i in range(faces.shape[1]):
        v1, v2, v3 = faces[:,i]
        p1, p2, p3 = verts.T[faces[:,i]]
        normal = np.cross(p2 - p1, p3 - p2)
        if not np.all(normal == 0):
            vertsn[:,v1] += normal
            vertsn[:,v2] += normal
            vertsn[:,v3] += normal
    for i in range(verts.shape[1]):
        if not np.all(verts[:,i] == 0):
            vertsn[:,i] = vertsn[:,i] / np.linalg.norm(vertsn[:,i])
    return vertsn

def shade_gouraud(vertsp: np.ndarray, vertsn: np.ndarray, vertsc: np.ndarray, bcoords: np.ndarray, cam_pos: np.ndarray, ka: float, kd: float, ks: float, n: int, lpos: np.ndarray, lint: np.ndarray, lamb: np.ndarray, X: np.ndarray) -> np.ndarray:
    # light calculation using center of mass
    vcolors = np.zeros((3,3))
    lint_total = np.concatenate((lint, lamb), axis=1)
    for i in range(3):
        color = vertsc[i]
        normal = vertsn[:,i].reshape((3,1))
        vcolors[i] = light(bcoords, normal, color, cam_pos, ka, kd, ks, n, lpos, lint_total)
  
    # application of gouraud shading
    if vertsp.shape[1]!=2:
        vertsp = vertsp.T
    Y = g_shading(X, vertsp, vcolors)
    return Y

def shade_phong(vertsp: np.ndarray, vertsn: np.ndarray, vertsc: np.ndarray, bcoords: np.ndarray, cam_pos: np.ndarray, ka: float, kd: float, ks: float, n: int, lpos: np.ndarray, lint: np.ndarray, lamb: np.ndarray, X: np.ndarray) -> np.ndarray:
    if vertsp.shape[1]!=2:
        vertsp = vertsp.T
    lint_total = np.concatenate((lint, lamb), axis=1)

    # Calculate the y scanning range    
    ymin, ymax = np.min(vertsp[:, 1]), np.max(vertsp[:, 1])
    
    # Initialize the result image
    updated_img = X.copy()

    normals = np.zeros(X.shape)
    colors = np.zeros(X.shape)
    for i in range(3):
        x,y = vertsp[i]
        colors[x][y] = vertsc[i]
        normals[x][y] = vertsn[i]
    
    # Find the edges of the triangle using the Bresenham Algorithm on every combination of vertices
    edges = [bresenham_line(vertsp[i], vertsp[(i + 1) % 3]) for i in range(3)]

    # Color in the edges and vertices
    for i in range(3):
        j = (i + 1) % 3
        # Handle excption of vertical line
        if vertsp[i][0] == vertsp[j][0]:
            if not vertsp[i][1] == vertsp[j][1]:
                for x, y in edges[i]:
                    normals[x][y] = [vector_interp(vertsp[i], vertsp[j], vertsn[c][i], vertsn[c][j], y, 2) for c in range(3)]
                    colors[x][y] = [vector_interp(vertsp[i], vertsp[j], vertsc[i][c], vertsc[j][c], y, 2) for c in range(3)]
                    updated_img[x][y] = light(bcoords, normals[x][y].reshape((3,1)), colors[x][y], cam_pos, ka, kd, ks, n, lpos, lint_total)
        else:
            for x, y in edges[i]:
                normals[x][y] = [vector_interp(vertsp[i], vertsp[j], vertsn[i][c], vertsn[j][c], x, 1) for c in range(3)]
                colors[x][y] = [vector_interp(vertsp[i], vertsp[j], vertsc[i][c], vertsc[j][c], x, 1) for c in range(3)]
                updated_img[x][y] = light(bcoords, normals[x][y].reshape((3,1)), colors[x][y], cam_pos, ka, kd, ks, n, lpos, lint_total)
    # Exit in case of double point (line painting)
    # if double_point:
    #     return updated_img

    # Concatenate the edges
    active_edges = np.concatenate(edges)

     # Scan all the y lines in the calculated range
    for y in range(ymin, ymax):
        # Move all the points with the same y into the current edges list
        current_edges = active_edges[active_edges[:, 1] == y][:, 0]
        
        # Skip the lines with only one point (vertex)
        if len(current_edges) <= 1:
            continue
        
        # Color in every pixel in the x scanning line using interpolation
        xmin, xmax = np.min(current_edges), np.max(current_edges)
        V1 = colors[xmin, y]
        V2 = colors[xmax, y]
        N1 = normals[xmin, y]
        N2 = normals[xmax, y]
        for x in range(xmin, xmax):
            normals[x][y] = [vector_interp([xmin, y], [xmax, y], N1[c], N2[c], x, 1) for c in range(3)]
            colors[x][y] = [vector_interp([xmin, y], [xmax, y], V1[c], V2[c], x, 1) for c in range(3)]
            updated_img[x][y] = light(bcoords, normals[x][y].reshape((3,1)), colors[x][y], cam_pos, ka, kd, ks, n, lpos, lint_total)

    return updated_img

def render_object(shader: str, focal: float, eye: np.ndarray, target: np.ndarray, up: np.ndarray, bg_color: np.ndarray, M: int, N: int, H: int, W: int, verts: np.ndarray, vert_colors: np.ndarray, faces: np.ndarray, ka: float, kd: float, ks: float, n: int, lpos: np.ndarray, lint: np.ndarray, lamb: np.ndarray) -> np.ndarray:
    # dimensions handling
    if verts.shape[0] != 3 or faces.shape[0] != 3:
        print("Wrong dimensions")
        return None

    # initialization
    img = np.ones((M, N, 3)) * bg_color
    normals = calculate_normals(verts, faces)

    # perspective projection and rasterization
    (R, c0) = lookat(eye, up, target)
    (pts_proj, depth) = perspective_project(verts, focal, R, c0)
    verts_pix = rasterize(pts_proj, W, H, N, M)
    
    # triangle depth sorting
    triangle_depth = np.mean(depth[faces.T], axis=1)
    sorted_depth_indexes = np.argsort(-triangle_depth)
    faces = faces[:, sorted_depth_indexes]

    if shader == 'gouraud':
        t_num = faces.shape[1]
        t_count = 0
        for t in faces.T:
            print(f"\r{int(100 * t_count / t_num)}%", end = "")
            t_count += 1
            current_verts = verts_pix[:,t]
            current_normals = normals[:,t]
            current_colors = vert_colors.T[t]   
            bcoords = np.mean(verts[:,t], axis=1).reshape((3,1))
            img = shade_gouraud(current_verts, current_normals, current_colors, bcoords, eye, ka, kd, ks, n, lpos, lint, lamb, img)
    elif shader == 'phong':
        t_num = faces.shape[1]
        t_count = 0
        for t in faces.T:
            print(f"\r{int(100 * t_count / t_num)}%", end = "")
            t_count += 1
            current_verts = verts_pix[:,t]
            current_normals = normals[:,t]
            current_colors = vert_colors.T[t]   
            bcoords = np.mean(verts[:,t], axis=1).reshape((3,1))
            img = shade_phong(current_verts, current_normals, current_colors, bcoords, eye, ka, kd, ks, n, lpos, lint, lamb, img)
    return img


# if __name__ == "__main__":
#     shader = 'gouraud'
#     focal = 1
#     eye = np.array([[0], [0], [0]])
#     target = np.array([[0], [0], [1]])
#     up = np.array([[0], [1], [0]])
#     bg_color = np.array([0, 0, 0])
#     M = 200
#     N = 200
#     H = 20
#     W = 20
#     verts = np.array([[0, 1, 2, 3, 4, 5, 6],
#                       [0, 1, 1, 3, 4, 5, 6],
#                       [1, 1, 1, 1, 1, 1, 1]])
#     vert_colors = np.array([[0, 0, 0],
#                             [1, 1, 1],
#                             [2, 2, 2],
#                             [3, 3, 3],
#                             [4, 4, 4],
#                             [5, 5, 5],
#                             [6, 6, 6]])
#     #faces = np.array([[0, 1, 2, 3, 4],
#                       #[1, 2, 3, 4, 5],
#                       #[2, 3, 4, 5, 6]])
#     faces = np.array([[0],
#                       [1],
#                       [2]])
#     ka = 0
#     kd = 0
#     ks = 0
#     n = 3
#     lpos = np.zeros((3,3))
#     lint = np.zeros((3,3))
#     lamb = np.zeros((3, 1))
#     render_object(shader, focal, eye, target, up, bg_color, M, N, H, W, verts, vert_colors, faces, ka, kd, ks, n, lpos, lint, lamb)
