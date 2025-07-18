# Code by Korel Ö.
# 18.07.2025


from utils.objLoader import objLoader
from utils.matrix import *
from PIL import Image
import math
import os

# saves file in "../output" directory with a time-based filename
def save_ppm(width, height, buffer, time):
    """
    Saves the framebuffer as a PPM image with the specified width, height, and time-based filename.
    """
    output_path = os.path.join(os.path.dirname(__file__), "output")
    output_file = os.path.join(output_path, f"image.{time}.ppm")
    os.makedirs(output_path, exist_ok=True)
    ppm_file = open(output_file, "w")

    # Write the PPM header
    header = f"P3\n{width} {height}\n255\n"
    ppm_file.write(header)

    # Write pixel data to the PPM file
    for index, value in enumerate(buffer):
        color_value = f"{value} "
        if (index + 1) % 3 == 0:
            color_value += "\n"
        ppm_file.write(color_value)

    print(f"file saved {time}")
    ppm_file.close()

# performs perspective division on a point `p` to project it onto the image plane.
def perspective_divide(p, image_plane_distance):
    return [
        image_plane_distance * p[0] / p[2],
        image_plane_distance * p[1] / p[2],
        image_plane_distance,
    ]

# converts a point from view space to raster coordinates
def view_to_raster(v, width, height):
    """
    Converts a point from view space (normalized device coordinates) to raster (pixel) coordinates.
    """
    raster_x = ((v[0] + 1) / 2) * width
    raster_y = ((1 - v[1]) / 2) * height
    return [round(raster_x), round(raster_y)]

# sets the color of a pixel in the raster
def set_raster_coordinate(x, y, r, g, b):
    offset = 3 * x + (width * 3) * y
    buffer[offset] = r
    buffer[offset + 1] = g
    buffer[offset + 2] = b

# multiplies two 4x4 matrices a and b
def m4_x_m4(a, b):
    return [
        # Row 1
        a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12],
        a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13],
        a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14],
        a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15],
        # Row 2
        a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12],
        a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13],
        a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14],
        a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15],
        # Row 3
        a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12],
        a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13],
        a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14],
        a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15],
        # Row 4
        a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12],
        a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13],
        a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14],
        a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15],
    ]

# from 3D Vector to 4D Vector
def vec3_to_vec4(v):
    return [v[0], v[1], v[2], 1.0]

#multiplies a 3D vector by a 4x4 matrix, returning the transformed 4D vector.
def mult_vec3_m4(v, m):
    v4 = vec3_to_vec4(v)
    return [
        v4[0] * m[0] + v4[1] * m[4] + v4[2] * m[8] + v4[3] * m[12],
        v4[0] * m[1] + v4[1] * m[5] + v4[2] * m[9] + v4[3] * m[13],
        v4[0] * m[2] + v4[1] * m[6] + v4[2] * m[10] + v4[3] * m[14],
        v4[0] * m[3] + v4[1] * m[7] + v4[2] * m[11] + v4[3] * m[15],
    ]

# rotates around the x-axis (3x3 matrix)
def rot_x(degrees):
    return [
        1, 0, 0,
        0, math.cos(degrees), -math.sin(degrees),
        0, math.sin(degrees), math.cos(degrees),
    ]

# rotates around the y-axis (3x3 matrix)
def rot_y(degrees):
    return [
        math.cos(degrees), 0, math.sin(degrees),
        0, 1, 0,
        -math.sin(degrees), 0, math.cos(degrees),
    ]

# rotates around the Z-axis (3x3 matrix)
def rot_z(degrees):
    return [
        math.cos(degrees), -math.sin(degrees), 0,
        math.sin(degrees), math.cos(degrees), 0,
        0, 0, 1,
    ]

#converts a 3x3 matrix to a 4x4 matrix by adding a translation row and column.
def m3_to_m4(m):
    return [
        m[0], m[1], m[2], 0,
        m[3], m[4], m[5], 0,
        m[6], m[7], m[8], 0,
        0, 0, 0, 1,
    ]

# cubic easing function for smooth animations.
def easeInOutCubic(x):
    if x < 0.5:
        return 4 * x * x * x
    else:
        return 1 - math.pow(-2 * x + 2, 3) / 2

# Subtracts two 2D vectors v1 and v2.
def v2_substract(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1]]

# computes determinant of 2x2 matrix
def determinant(m):
    return m[0] * m[3] - m[1] * m[2]

# Creates a 2x2 matrix from two 2D vectors v1 and p1.
def makematrix(v1, p1):
    return [v1[0], p1[0], v1[1], p1[1]]

# Computes the axis-aligned bounding box (AABB) for a triangle defined by raster coordinates.
def bbox(face_vertices_raster_coordinates, width, height):
    """
    Computes the axis-aligned bounding box (AABB) for a triangle defined by raster coordinates.
    Returns [[min_x, min_y], [max_x, max_y]].
    """
    bboxmin = [width, height]
    bboxmax = [0, 0]
    for v in face_vertices_raster_coordinates:
        bboxmin[0] = min(bboxmin[0], v[0])
        bboxmin[1] = min(bboxmin[1], v[1])
        bboxmax[0] = max(bboxmax[0], v[0])
        bboxmax[1] = max(bboxmax[1], v[1])
    return [bboxmin, bboxmax]

# Dot product of two 3D vectors
def v3_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

# Compute the length (magnitude) of a 3D vector
def v3_length(v):
    return math.sqrt(v3_dot(v, v))

# Normalize a 3D vector to unit length
def v3_normalize(v):
    length = v3_length(v)
    if not length:
        length = 1
    inv_length = 1 / length
    return [v[0] * inv_length, v[1] * inv_length, v[2] * inv_length]

# Compute the cross product of two 3D vectors
def v3_cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]

# Subtract two 3D vectors: a - b
def v3_substract(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


# Load texture and 3D model
tex = Image.open("./woche9/obj/lamp.jpg")
o = objLoader("./woche9/obj/lamp.obj")
tex_rgb = tex.convert("RGB")
tex_width, tex_height = tex.size

# Image resolution in pixels
width = 800
height = 800
buffer_length = width * height

# Initialize color and depth buffers
buffer = [10] * buffer_length * 3
depth_buffer = [-9999.99] * buffer_length

# Transformation matrix for translating the object along the Z axis
translation = [
    1, 0, 0, 0, 
    0, 1, 0, 0, 
    0, 0, 1, 0, 
    0, 0, -6, 1,
]

# Direction vector for the light source
light = [0, 1, 1]

# Number of frames to render (for animation)
frames = 50

# Loop over animation frames
for t in range(0, frames):
    # Reset buffers for each frame
    buffer = [10] * buffer_length * 3
    depth_buffer = [-9999.99] * buffer_length

    # Compute animation easing and rotation
    normalized_t = t / frames
    eased_value = easeInOutCubic(normalized_t)
    angle = (360 / frames) * (eased_value * frames)
    radians = math.radians(angle) # Convert angle to radians

    # Compute rotation matrix around Y-axis
    rotationy = m3_to_m4(rot_y(radians))

    # Combine rotation and translation into a single transformation matrix
    combined = m4_x_m4(rotationy, translation)

    # Compute inverse transpose matrix for transforming normals
    combined_i = m4_transpose(m4_invert(combined))

    # Iterate through each triangle face of the object
    for index, value in enumerate(o.faceVertexIndices[::3]):

        v_normals = []  # List to store per-vertex normals

        # Get vertex indices of the triangle
        v1 = o.faceVertexIndices[index * 3]
        v2 = o.faceVertexIndices[(index * 3) + 1]
        v3 = o.faceVertexIndices[(index * 3) + 2]

        # Compute flat vertex positions for each triangle vertex
        v_index1 = (v1 - 1) * 3
        v_index2 = (v2 - 1) * 3
        v_index3 = (v3 - 1) * 3

        v_faces = [
            o.vertices[v_index1], o.vertices[v_index1 + 1], o.vertices[v_index1 + 2],
            o.vertices[v_index2], o.vertices[v_index2 + 1], o.vertices[v_index2 + 2],
            o.vertices[v_index3], o.vertices[v_index3 + 1], o.vertices[v_index3 + 2]
        ]

        # Retrieve texture coordinates for the triangle vertices
        num_uv1 = o.faceUVIndices[index * 3]
        num_uv2 = o.faceUVIndices[(index * 3) + 1]
        num_uv3 = o.faceUVIndices[(index * 3) + 2]

        uv1_index = (num_uv1 - 1) * 2
        uv2_index = (num_uv2 - 1) * 2
        uv3_index = (num_uv3 - 1) * 2

        uv1 = [o.uv[uv1_index], o.uv[uv1_index + 1]]
        uv2 = [o.uv[uv2_index], o.uv[uv2_index + 1]]
        uv3 = [o.uv[uv3_index], o.uv[uv3_index + 1]]

        # Retrieve normals for each triangle vertex
        n1 = o.faceNormalsIndices[index * 3]
        n2 = o.faceNormalsIndices[(index * 3) + 1]
        n3 = o.faceNormalsIndices[(index * 3) + 2]

        n_index1 = (n1 - 1) * 3
        n_index2 = (n2 - 1) * 3
        n_index3 = (n3 - 1) * 3

        normal1 = [o.normals[n_index1], o.normals[n_index1 + 1], o.normals[n_index1 + 2]]
        normal2 = [o.normals[n_index2], o.normals[n_index2 + 1], o.normals[n_index2 + 2]]
        normal3 = [o.normals[n_index3], o.normals[n_index3 + 1], o.normals[n_index3 + 2]]

        v_normals.extend(normal1)
        v_normals.extend(normal2)
        v_normals.extend(normal3)

        raster_coordinates = []  # List of triangle vertex positions in raster space
        v_t = []                # List of triangle vertices transformed to view space
        v_normals_t = []        # List of normals transformed to view space

        # Transform each triangle vertex and normal
        for index, val in enumerate(v_faces[::3]):
            start_index = index * 3

            v = [
                v_faces[start_index],
                v_faces[start_index + 1],
                v_faces[start_index + 2]
            ]
            # Apply transformation to vertex
            v_transformed = mult_vec3_m4(v, combined)
            v_t.append(v_transformed)

            # Apply transformation to normal
            n = [
                v_normals[start_index],
                v_normals[start_index + 1],
                v_normals[start_index + 2]
            ]
            normal_transformed = mult_vec3_m4(n, combined_i)
            v_normals_t.append(normal_transformed)

            # Convert view-space coordinates to raster coordinates
            screen_space_point = perspective_divide(v_transformed, -1)
            raster_point = view_to_raster(screen_space_point, width, height)
            raster_coordinates.extend(raster_point)

        # Calculate face normal using cross product of triangle edges
        v3_edge_ab = v3_substract(v_t[1], v_t[0])
        v3_edge_ac = v3_substract(v_t[2], v_t[0])
        face_normal = v3_cross(v3_normalize(v3_edge_ab), v3_normalize(v3_edge_ac))

        # Extract individual vertex raster coordinates
        a = [raster_coordinates[0], raster_coordinates[1]]
        b = [raster_coordinates[2], raster_coordinates[3]]
        c = [raster_coordinates[4], raster_coordinates[5]]

        # Compute triangle edge vectors in 2D for rasterization
        v_edge_ab = v2_substract(b, a)
        v_edge_bc = v2_substract(c, b)
        v_edge_ca = v2_substract(a, c)
        v_edge_ac = v2_substract(c, a)

        m2_face = [
            v_edge_ab[0], v_edge_ab[1],
            v_edge_ac[0], v_edge_ac[1]
        ]

        # Compute area of the triangle using the determinant
        face_area = determinant(m2_face)

        # Compute the bounding box of the triangle
        [bbox_min, bbox_max] = bbox([a, b, c], width, height)

        # Loop over every pixel inside the bounding box
        for y in range(bbox_min[1], bbox_max[1]):
            for x in range(bbox_min[0], bbox_max[0]):
                p = [x, y]

                # Compute vectors from vertices to current pixel p
                v_ap = v2_substract(p, a)
                v_bp = v2_substract(p, b)
                v_cp = v2_substract(p, c)

                # Compute determinants for inside-triangle test
                det_a = determinant(makematrix(v_edge_ab, v_ap))
                det_b = determinant(makematrix(v_edge_bc, v_bp))
                det_c = determinant(makematrix(v_edge_ca, v_cp))

                # Skip pixels outside the triangle or on the degenerate line
                if det_a == 0 and det_b == 0 and det_c == 0:
                    continue

                # If the pixel lies inside the triangle
                if det_a <= 0 and det_b <= 0 and det_c <= 0:

                    # Compute barycentric weights
                    weight_a = det_b / face_area
                    weight_b = det_c / face_area
                    weight_c = 1 - weight_a - weight_b

                    # Interpolate UV coordinates using barycentric weights
                    u = uv1[0] * weight_a + uv2[0] * weight_b + uv3[0] * weight_c
                    v = uv1[1] * weight_a + uv2[1] * weight_b + uv3[1] * weight_c

                    u = u % 1.0
                    v = v % 1.0

                    # Interpolate 3D position in view space (for depth and lighting)
                    interpolated_z = v_t[0][2] * weight_a + v_t[1][2] * weight_b + v_t[2][2] * weight_c
                    v_i = [0, 0, interpolated_z]  # Only z used for depth

                    # Interpolate transformed normals
                    interpolated_n_x = v_normals_t[0][0] * weight_a + v_normals_t[1][0] * weight_b + v_normals_t[2][0] * weight_c
                    interpolated_n_y = v_normals_t[0][1] * weight_a + v_normals_t[1][1] * weight_b + v_normals_t[2][1] * weight_c
                    interpolated_n_z = v_normals_t[0][2] * weight_a + v_normals_t[1][2] * weight_b + v_normals_t[2][2] * weight_c
                    v_normal = [interpolated_n_x, interpolated_n_y, interpolated_n_z]

                    # Sample the texture at the interpolated UV coordinates
                    tex_x = int(u * (tex_width-1))
                    tex_y = int((v * -1 + 1) * (tex_height-1))
                    tex_r, tex_g, tex_b = tex_rgb.getpixel((tex_x, tex_y))

                    # Compute lighting using the interpolated normal
                    light_incident = v3_substract(light, v_i)
                    dot = v3_dot(v3_normalize(light_incident), v3_normalize(v_normal))

                    # Apply lighting to the texture color
                    color_r = int(tex_r * dot)
                    color_g = int(tex_g * dot)
                    color_b = int(tex_b * dot)

                    # Clamp final color values to 0-255 range
                    color_r = max(0, min(255, color_r))
                    color_g = max(0, min(255, color_g))
                    color_b = max(0, min(255, color_b))

                    # Update depth buffer and color buffer if the pixel is closer to the camera
                    if depth_buffer[x + (y * width)] < interpolated_z:
                        depth_buffer[x + (y * width)] = interpolated_z
                        set_raster_coordinate(
                            p[0],
                            p[1],
                            color_r,
                            color_g,
                            color_b
                        )
    # Save the rendered frame as an image
    save_ppm(width, height, buffer, t)

