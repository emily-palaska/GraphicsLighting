import numpy as np
from light import light
from perspective_project import perspective_project
from lookat import lookat
from g_shading import *
from rasterize import rasterize
import matplotlib.pyplot as plt

def calculate_normals(verts:np.ndarray, faces:np.ndarray) -> np.ndarray:
    """
    Calculates the normal vectors corresponding to every vertex

    Args:
        - verts: 3d coordinates of every given point
        - faces: triangle vertices, corresponding to indices in the verts array 
    
    Returns:
        - vertsn: the calculated normals at every vertex point
    """
        
    # Summation matrix initialization    
    vertsn = np.zeros(verts.shape)

    # Triangle iteration
    for i in range(faces.shape[1]):
        v1, v2, v3 = faces[:,i]
        p1, p2, p3 = verts.T[faces[:,i]]
        normal = np.cross(p2 - p1, p3 - p2)
        # Zero vector handling
        if not np.all(normal == 0):
            vertsn[:,v1] += normal
            vertsn[:,v2] += normal
            vertsn[:,v3] += normal
    # Summation matrix normalization
    for i in range(verts.shape[1]):
        if not np.all(verts[:,i] == 0):
            vertsn[:,i] = vertsn[:,i] / np.linalg.norm(vertsn[:,i])
    return vertsn

def shade_gouraud(vertsp: np.ndarray, vertsn: np.ndarray, vertsc: np.ndarray, bcoords: np.ndarray, cam_pos: np.ndarray, ka: float, kd: float, ks: float, n: int, lpos: np.ndarray, lint: np.ndarray, lamb: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Applies coloring with the gouraud shading technique and the lighting model
    
    Args:
        - vertsp: 2d coordinates of vertices
        - vertsn: 3d normal coordinates corresponding to every vertex
        - vertsc: RGB values of color corresponding to every vertex
        - bcoords: 3d coordinates of the vertices centroid 
        - cam_pos: 3d coordinates of camera position
        - ka: ambient light constant
        - kd: diffuse light constant
        - ks: specular light constant
        - n: phong constant
        - lpos: light source positions
        - lint: light source intensities
        - lamb: ambient light intensity
        - X: image with some triangles already shaded
    
    Returns:
        - Y: the image with all the triangles shaded
    """
    
    # Light calculation using center of mass
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
    """
    Applies coloring with the phong shading technique and the lighting model
    
    Args:
        - vertsp: 2d coordinates of vertices
        - vertsn: 3d normal coordinates corresponding to every vertex
        - vertsc: RGB values of color corresponding to every vertex
        - bcoords: 3d coordinates of the vertices centroid 
        - cam_pos: 3d coordinates of camera position
        - ka: ambient light constant
        - kd: diffuse light constant
        - ks: specular light constant
        - n: phong constant
        - lpos: light source positions
        - lint: light source intensities
        - lamb: ambient light intensity
        - X: image with some triangles already shaded
    
    Returns:
        - Y: the image with all the triangles shaded
    """
    
    # Dimension handling
    if vertsp.shape[1]!=2:
        vertsp = vertsp.T
    lint_total = np.concatenate((lint, lamb), axis=1)
    double_point = False

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
                double_point = True
        else:
            for x, y in edges[i]:
                normals[x][y] = [vector_interp(vertsp[i], vertsp[j], vertsn[i][c], vertsn[j][c], x, 1) for c in range(3)]
                colors[x][y] = [vector_interp(vertsp[i], vertsp[j], vertsc[i][c], vertsc[j][c], x, 1) for c in range(3)]
                updated_img[x][y] = light(bcoords, normals[x][y].reshape((3,1)), colors[x][y], cam_pos, ka, kd, ks, n, lpos, lint_total)

    # Double point handling
    if double_point:
        return updated_img
    
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
    """"
    Creates an image and handles all the calculations for the depiction of an object

    Args:
        - shader: type of shading ('gouraud'/'phong')
        - focal: camera parameter
        - eye: camera position
        - target: camera target
        - up: camera up vector
        - bg_color: background color, numpy vector with the RGB values
        - M: camera plane height
        - N: camera plane width
        - H: image height
        - W: image width
        - verts: 3d coordinates of vertices
        - vert_colors: RBG values of colors (indices corresponding to verts)
        - faces: triangle combinations to paint (indices corresponding to verts)
        - ka: ambient light constant
        - kd: diffuse light constant
        - ks: specular light constant
        - n: phong constant
        - lpos: light source positions
        - lint: light source intensities
        - lamb: ambient light intensity

        Returns:
        - img: image with the rendered object colored with specific shader trechnique
    """
    
    # Dimension handling
    if verts.shape[0] != 3 or faces.shape[0] != 3:
        print("Wrong dimensions")
        return None

    # Initialization
    img = np.ones((M, N, 3)) * bg_color
    normals = calculate_normals(verts, faces)

    # Perspective projection and rasterization
    (R, c0) = lookat(eye, up, target)
    (pts_proj, depth) = perspective_project(verts, focal, R, c0)
    verts_pix = rasterize(pts_proj, W, H, N, M)
    
    # Triangle depth sorting
    triangle_depth = np.mean(depth[faces.T], axis=1)
    sorted_depth_indexes = np.argsort(-triangle_depth)
    faces = faces[:, sorted_depth_indexes]

    # Shader technique application by iterating all triangles
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