import numpy as np
from vector_interp import vector_interp

# Function that performs the Bresenham Algorithm between two points 
#
# Input: 
#   start: the starting point of the line
#   end: the ending point of the line
#
# Output:
#   pixels: a list of all the pixels in the line connecting the two points
def bresenham_line(start, end):
    # Initialize algorithm by calculating the step and error variables
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    # Loop to decide the next pixel until ending point is reached
    pixels = []
    while True:
        pixels.append([x1, y1])
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return np.array(pixels)

# Function that fills a given triangle with the Gouraud shading technique
# Given that the algorithm only handles triangles, non-convex polygons are not considered in the implentation
#
# Input:
#   img: the given canvas with pre-existing shapes
#   vertices: the three points defining the triangle to be shaded
#   vcolors: the colors od the three vertices
#
# Output:
#   update_img: the new canvas with the filled triangle
def g_shading(img, vertices, vcolors):    
    # Calculate the y scanning range
    ymin, ymax = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    
    # Initialize the result image
    updated_img = img.copy()   
    
    # Find the edges of the triangle using the Bresenham Algorithm on every combination of vertices
    edges = [bresenham_line(vertices[i], vertices[(i + 1) % 3]) for i in range(3)]

    # Color in the edges and vertices
    for i in range(3):
        j = (i + 1) % 3
        # Handle excption of vertical line
        if vertices[i][0] == vertices[j][0]:
            for x, y in edges[i]:
                updated_img[x][y] = [vector_interp(vertices[i], vertices[j], vcolors[i][c], vcolors[j][c], y, 2) for c in range(3)]
        else:
            for x, y in edges[i]:
                updated_img[x][y] = [vector_interp(vertices[i], vertices[j], vcolors[i][c], vcolors[j][c], x, 1) for c in range(3)]
        updated_img[vertices[i][0]][vertices[i][1]] = vcolors[i]  
    
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
        V1 = updated_img[xmin, y]
        V2 = updated_img[xmax, y]
        for x in range(xmin, xmax):
            gouraud_color = [vector_interp([xmin, y], [xmax, y], V1[c], V2[c], x, 1) for c in range(3)]
            updated_img[x][y] = gouraud_color
    return updated_img