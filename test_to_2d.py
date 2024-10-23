import open3d as o3d
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from shapely.geometry import MultiPoint, box
import numpy as np
from typing import List, Tuple, Union


def view_points(points: np.ndarray, view: np.ndarray) -> np.ndarray:
# points就是我们的3d框的8个角坐标；view就是我们所需要用到对应相机的相机内参。
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    # viewpad = np.eye(4)
    # viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(view, points)
    points = points[:3, :]

    # if normalize:
    points = points[:2, :] / points[2, :]
    # points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int]) -> Union[Tuple[float, float, float, float], None]:
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull  # 多边形
    img_canvas = box(0, 0, imsize[1], imsize[0])  # 图像的画布 box（minx,miny,maxx,maxy）左上右下

    if polygon_from_2d_box.intersects(img_canvas):  # 如果相交
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        # print(min_x, min_y, max_x, max_y)
        return min_x, min_y, max_x, max_y
        # return min_y, min_x, max_y, max_x
    else:
        return None


def transform_to_camera(extrinsic_matrix, location, R0_rect):
    # R = self.extrinsic_matrix[:, :3]
    # T = self.extrinsic_matrix[:, 3]

    # location = np.array(location).reshape(1,3)

    # # 1*3
    # location_cam = location.dot(R.T) + T
    location_homo = np.append(location, 1)

    extrnsic_matrix_homo = np.vstack([extrinsic_matrix, np.array([0, 0, 0, 1])])
    location_cam_unrect = np.dot(extrnsic_matrix_homo, location_homo)
    
    R0_rect = np.hstack([R0_rect, np.zeros((3, 1))])
    R0_rect = np.vstack([R0_rect, np.array([0, 0, 0, 1])])
    location_cam = np.dot(R0_rect, location_cam_unrect)
    # R_z = np.array([
    #     [0, -1, 0],
    #     [1, 0, 0],
    #     [0, 0, 1]
    # ]) 
    # location_cam = np.dot(R_z, location_cam[:3])

    return location_cam[:3]


def get_data(line):
    parts = line.strip().split()
    dimension = [float(parts[8]), float(parts[9]), float(parts[10])]
    position = [float(parts[11]), float(parts[12]), float(parts[13])]
    rotation_y = float(parts[14])
    return dimension, position, rotation_y

# intrinsic_matrix = np.array([[718.3351, 0, 600.3891, 0],
#                 [0, 718.3351, 181.5122, 0],
#                 [0, 0, 1, 0]], dtype=float)

# extrinsic_matrix = np.array([[0.007755449, -0.9999694, -0.001014303, -0.007275538],
#                 [0.002294056, 0.001032122, -0.9999968, -0.06324057],
#                 [0.9999673, 0.007753097, 0.002301990, -0.2670414]], dtype=float)

# R0_rect = np.array([
#     [0.9999478, 0.009791707, -0.002925305],
#     [-0.009806939, 0.9999382, -0.005238719],
#     [0.002873828, 0.005267134, 0.9999820]
# ])

intrinsic_matrix = np.array([[969.94239058, 0, 956.28949532, 0],
     [0, 971.9671513, 556.70141901, 0],
     [0, 0, 1, 0]], dtype=float)

extrinsic_matrix = np.array([ [-0.0114814, -0.999882, 0.0101998, 0.23438],
    [0.0421775, -0.0106757, -0.999053, -0.452339],
    [0.999045, -0.0110404, 0.0422951, -0.543866]], dtype=float)

R0_rect = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

kitti_file_path = r"C:\Xin Yutong\5 MA\train_data\2024_07_08-16_47_49\training\label_2\000000.txt"
image_file_path = r"C:\Xin Yutong\5 MA\train_data\2024_07_08-16_47_49\training\image_2\000000.png"
labeled_image_folder_path = r"C:\Xin Yutong\5 MA\train_data\2024_07_08-16_47_49\trainingimage_labeled"
labeled_image_name = '000000_labeled.png'
labeled_image_file_path = os.path.join(labeled_image_folder_path, labeled_image_name)
# calib_file_path = r"C:\Xin Yutong\5 MA\labeled_data\to_KITTI\test\000005.txt"
image = cv2.imread(image_file_path)
imsize = image.shape[:2]

# calib = pd.read_csv(calib_file_path)

with open(kitti_file_path, 'r') as file:


    for ll in file:


        dimension, position, rotation_y = get_data(ll)

        w = dimension[0]
        l = dimension[1]
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
        
        # transfer to camera
        
        # bbox_corners_camera = []

        # for i in bbox_corners:
        #     bbox_corners_camera.append(transform_to_camera(extrinsic_matrix, i))
        #     # bbox_corners_camera = transform_to_camera(extrinsic_matrix, i)
        #     # min_x, min_y, max_x, max_y = post_process_coords(bbox_corners_camera, imsize)
        
        # bbox_corners_camera_array = np.array(bbox_corners_camera)
        

        bbox_corners_camera = np.zeros((8,3))
        for i in range(bbox_corners.shape[0]):
            bbox_corners_camera[i, :] = transform_to_camera(extrinsic_matrix, bbox_corners[i], R0_rect)


        points = view_points(bbox_corners_camera.T, intrinsic_matrix)
        print(points)

        if post_process_coords(points.T, imsize) is not None:
            min_x, min_y, max_x, max_y = post_process_coords(points.T, imsize)
        else: 
            continue


        color = (0, 255, 0) 
        thickness = 2


        image_with_rectangle = cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color, thickness)
        

        cv2.imshow("Image with Rectangle", image_with_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(labeled_image_file_path, image)
   


