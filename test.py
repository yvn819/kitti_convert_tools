import open3d as o3d
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from PIL import Image

np.random.seed(42)

def get_data(line):
    parts = line.strip().split()
    dimension = [float(parts[8]), float(parts[9]), float(parts[10])]
    position = [float(parts[11]), float(parts[12]), float(parts[13])]
    rotation_y = float(parts[14])
    return dimension, position, rotation_y

def rotate_points_around_center(points, centre_point, yaw_angle):
    
    # Define the rotation matrix for yaw (rotation around z-axis)
    cos_yaw = np.cos(yaw_angle)
    sin_yaw = np.sin(yaw_angle)

    # R_yaw = np.array([
    #     [cos_yaw, 0, sin_yaw],
    #     [0, 1, 0],
    #     [-sin_yaw, 0, cos_yaw]
    # ])
    R_yaw = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]])
    
    R_z = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])    
    R_x = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    R_y = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    # R_yaw = np.eye(3)
    
    # Compute the center of the object
    center = centre_point
    
    # Translate the points so that the center is at the origin
    points_centered = points - center
    
    # Rotate the centered points
    rotated_points_centered = np.dot(R_yaw, points_centered)
    # rotated_points_centered = np.dot(R_z, rotated_points_centered)
    # rotated_points_centered = np.dot(R_x, rotated_points_centered)
    # rotated_points_centered = np.dot(R_y, rotated_points_centered)
    
    # Translate the points back to the original position
    rotated_points = rotated_points_centered + center
    
    return rotated_points

def read_calib_file(calib_file_path):
    calib_data = {}
    with open(calib_file_path, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.split(':', 1)
                calib_data[key] = np.array([float(x) for x in value.split()])
    return calib_data

def get_calib_matrices(calib_data):
    # Get P0, Tr_velo_to_cam, R0_rect matrices
    P0 = calib_data['P0'].reshape(3, 4)
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
    R0_rect = calib_data['R0_rect'].reshape(3, 3)
    return P0, Tr_velo_to_cam, R0_rect


frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])

kitti_file_path = r"C:\Xin Yutong\5 MA\train_data\2024_07_08-16_47_49\training\label_2_org\000000.txt"
image_file_path = r"C:\Xin Yutong\5 MA\train_data\2024_07_08-16_47_49\training\image_2\000000.png"
labeled_image_folder_path = r"C:\Xin Yutong\5 MA\train_data\2024_07_08-16_47_49\training\training\image_labeled"
labeled_image_name = '000000_labeled.png'
labeled_image_file_path = os.path.join(labeled_image_folder_path, labeled_image_name)
calib_file_path = r"C:\Xin Yutong\5 MA\train_data\2024_07_08-16_47_49\training\calib\000000.txt"
image = cv2.imread(image_file_path)



calib_data = read_calib_file(calib_file_path)
# P2, Tr, R0 = get_calib_matrices(calib_data)


P2 = np.array([[969.94239058, 0, 956.28949532, 0],
                [0, 971.9671513, 556.70141901, 0],
                [0, 0, 1, 0]], dtype=float)

k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
Tr = np.array([[-0.0114814, -0.999882, 0.0101998, 0.23438],
                [0.0421775, -0.0106757, -0.999053, -0.452339],
                [0.999045, -0.0110404, 0.0422951, -0.543866
]], dtype=float)

R0 =  np.eye(3)

# 3D bbox
with open(kitti_file_path, 'r') as file:


    for ll in file:


        dimension, position, rotation_y = get_data(ll)

        # w = dimension[0]
        # l = dimension[1]
        # h = dimension[2]
        
        l = dimension[0]
        w = dimension[1]
        h = dimension[2]   


        x = position[0]
        y = position[1]
        z = position[2]

        rot_y=rotation_y

        bbox_corners = np.array([
                [x-l/2, y-w/2, z-h/2],
                [x+l/2, y-w/2, z-h/2],
                [x+l/2, y-w/2, z+h/2],
                [x-l/2, y-w/2, z+h/2],
                [x-l/2, y+w/2, z-h/2],
                [x+l/2, y+w/2, z-h/2],
                [x+l/2, y+w/2, z+h/2],
                [x-l/2, y+w/2, z+h/2],
            ])
        # print(bbox_corners)
        print(l, w, h)
        points=[]

        for i in bbox_corners:

            rotated_points = rotate_points_around_center(i, [x, y, z], rot_y)
            points.append(rotated_points)
            # points.append(i)

        


        vertices=[]

        for i in points:

            # Convert R0_rect to 4x4 matrix
            R0_rect_4x4 = np.hstack([R0, np.zeros((3, 1))])
            R0_rect_4x4 = np.vstack([R0_rect_4x4, np.array([0, 0, 0, 1])])

            # Point in Velodyne coordinates
            P_velo = np.array([i[0], i[1], i[2]])
            # print(P_velo)
            # print(i)

            # Transform to homogeneous coordinates
            P_homo_velo = np.append(P_velo, 1)

            # # Transform to unrectified camera coordinates
            Tr_4x4 = np.vstack([Tr, np.array([0, 0, 0, 1])])
            P_cam_unrect = np.dot(Tr_4x4, P_homo_velo)
            # # print(P_cam_unrect)

            # # Rectify the camera coordinates
            # P_cam_rect = np.dot(R0_rect_4x4, P_cam_unrect)

            # Project onto image plane
            P_image = np.dot(P2, P_cam_unrect)
            # P_image = np.dot(P2, P_homo_velo)
            # print(P_image)

            # Convert to pixel coordinates
            # u = P_image[0] / P_image[2]
            # v = P_image[1] / P_image[2]

            # print(f"The point in image coordinates is at ({u}, {v})")

            u = int(round(P_image[0] / P_image[2]))
            v = int(round(P_image[1] / P_image[2]))

            vertices.append([u,v])

            # Draw the point on the image
            color = (0, 255, 0) # Color of the point (red in BGR format)
            radius = 3 # Radius of the circle
            cv2.circle(image, (u, v), radius, color, -1) # -1 fills the circle

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical lines
        ]

        for line in lines:
            start = tuple(vertices[line[0]])
            end = tuple(vertices[line[1]])
            color = (0, 255, 255)  # Green color in BGR
            thickness = 2
            cv2.line(image, start, end, color, thickness)

'''# 2d bbox
with open(kitti_file_path, 'r') as file:
    for ll in file:
        dimension, position, rotation_y = get_data(ll)

        h = dimension[0]
        w = dimension[1]
        l = dimension[2]

        x = position[0]
        y = position[1]
        z = position[2]

        bbox_corners = np.array([
            [x - l/2, y - w/2, z - h/2],
            [x + l/2, y - w/2, z - h/2],
            [x + l/2, y - w/2, z + h/2],
            [x - l/2, y - w/2, z + h/2],
            [x - l/2, y + w/2, z - h/2],
            [x + l/2, y + w/2, z - h/2],
            [x + l/2, y + w/2, z + h/2],
            [x - l/2, y + w/2, z + h/2],
        ])

        edges = {
            'length': l,
            'width': w,
            'height': h
        }

        max_edge = max(edges, key=edges.get)
        max_length = edges[max_edge]

        # h max
        # bbox_corners_cut = np.array([
        #     [x - l/2, y, z - h/2],
        #     [x + l/2, y, z - h/2],
        #     [x + l/2, y, z + h/2],
        #     [x - l/2, y, z + h/2],
        # ])

        # w max
        # bbox_corners_cut = np.array([
        #     [x - l/2, y - w/2, z],
        #     [x + l/2, y - w/2, z],
        #     [x + l/2, y + w/2, z],
        #     [x - l/2, y + w/2, z],
        # ])


        if max_edge == 'length':

            cut_plane = [x, y, z] 
            bbox_corners_cut = np.array([
                [x, y - w/2, z - h/2],
                [x, y - w/2, z + h/2],
                [x, y + w/2, z + h/2],
                [x, y + w/2, z - h/2]
            ])
        elif max_edge == 'height':

            cut_plane = [x, y, z] 
            bbox_corners_cut = np.array([
                [x - l/2, y, z - h/2],
                [x + l/2, y, z - h/2],
                [x + l/2, y, z + h/2],
                [x - l/2, y, z + h/2],
            ])
        else:

            cut_plane = [x, y, z]  
            bbox_corners_cut = np.array([
                [x - l/2, y - w/2, z],
                [x + l/2, y - w/2, z],
                [x + l/2, y + w/2, z],
                [x - l/2, y + w/2, z],
            ])


        points = []
        for i in bbox_corners_cut:
            rotated_points = rotate_points_around_center(i, [x, y, z], rotation_y)
            points.append(rotated_points)

        vertices = []


        for i in points:
            P_velo = np.array([i[0], i[1], i[2]])


            P_homo_velo = np.append(P_velo, 1)


            Tr_4x4 = np.vstack([Tr, np.array([0, 0, 0, 1])])
            P_cam_unrect = np.dot(Tr_4x4, P_homo_velo)


            P_image = np.dot(P2, P_cam_unrect)


            u = int(round(P_image[0] / P_image[2]))
            v = int(round(P_image[1] / P_image[2]))

            vertices.append([u, v])

        vertices = np.array(vertices)


        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            # [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            # [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
        ]

        for line in lines:
            start = tuple(vertices[line[0]])
            end = tuple(vertices[line[1]])
            color = (0, 255, 255)  # 颜色 (黄)
            thickness = 2
            cv2.line(image, start, end, color, thickness)
'''

# Display the image
cv2.imshow('Image with Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(labeled_image_file_path, image)

# plt.figure(figsize=(15,8))
# plt.imshow(image)