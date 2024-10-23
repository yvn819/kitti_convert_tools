import os
import numpy as np
import cv2
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

class KITTIBBoxDrawer:
    def __init__(self, intrinsic_matrix, extrinsic_matrix, R0_rect):
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.R0_rect = R0_rect
    
    @staticmethod
    def read_data(file_path: str):
        with open(file_path, 'r') as file:
            data = file.readlines()
        return data
    
    @staticmethod
    def parse_line(line: str):
        parts = line.strip().split()
        # print(parts)
        # lidar
        label = parts[0]
        dimension = [float(parts[8]), float(parts[9]), float(parts[10])]
        # camera
        # dimension = [float(parts[9]), float(parts[10]), float(parts[8])]
        position = [float(parts[11]), float(parts[12]), float(parts[13])]
        rotation_y = float(parts[14])
        return label, dimension, position, rotation_y
    

    def position_transform_to_camera(self, location):
        # R = self.extrinsic_matrix[:, :3]
        # T = self.extrinsic_matrix[:, 3]

        # location = np.array(location).reshape(1,3)

        # # 1*3
        # location_cam = location.dot(R.T) + T
        location_homo = np.append(location, 1)
        extrnsic_matrix_homo = np.vstack([self.extrinsic_matrix, np.array([0, 0, 0, 1])])
        location_cam = np.dot(extrnsic_matrix_homo, location_homo)
        # print(location_cam[:3])
        return location_cam[:3]


    def transform_to_camera(self, location, position, rotation_y):

        rotation_matrix_z = np.array([
            [np.cos(rotation_y), -np.sin(rotation_y), 0],
            [np.sin(rotation_y), np.cos(rotation_y), 0],
            [0, 0, 1]
        ])

        location = np.array(location)
        position = np.array(position)

        location_centered = location - position
        rotated_location_centered = np.dot(rotation_matrix_z, location_centered)
        rotated_location = rotated_location_centered + position

        location_homo = np.append(rotated_location, 1)
        extrnsic_matrix_homo = np.vstack([self.extrinsic_matrix, np.array([0, 0, 0, 1])])
        location_cam_unrect = np.dot(extrnsic_matrix_homo, location_homo)
        
        R0_rect_homo = np.hstack([self.R0_rect, np.zeros((3, 1))])
        R0_rect_homo = np.vstack([R0_rect_homo, np.array([0, 0, 0, 1])])
        location_cam = np.dot(R0_rect_homo, location_cam_unrect)
        
        return location_cam[:3]
    
    @staticmethod
    def view_points(points: np.ndarray, view: np.ndarray) -> np.ndarray:
        assert points.shape[0] == 3
        nbr_points = points.shape[1]
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(view, points)
        points = points[:3, :]
        points = points[:2, :] / points[2, :]
        return points
    
    @staticmethod
    def post_process_coords(corner_coords: List, imsize: Tuple[int, int]) -> Union[Tuple[float, float, float, float], None]:
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[1], imsize[0])
        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])
            
            return min_x, min_y, max_x, max_y
        else:
            return None
    
    def write_bbox_coord_and_alpha(self, label_file_path, lines, bbox_coords_list, alpha_list, scale_list, position_list):
        """Write updated bounding box coordinates to the label file."""
        with open(label_file_path, 'w') as file:
            for line, bbox_coords, alpha, scale, position in zip(lines, bbox_coords_list, alpha_list, scale_list, position_list):
                parts = line.strip().split()
                if len(parts) >= 8 and bbox_coords:
                    min_x, min_y, max_x, max_y = bbox_coords
                    parts[3] = str(alpha)
                    parts[4] = str(min_x)
                    parts[5] = str(min_y)
                    parts[6] = str(max_x)
                    parts[7] = str(max_y)
                    ######## 0: h - 1:w - 2:l
                    parts[8] = str(scale[0])
                    parts[9] = str(scale[2])
                    parts[10] = str(scale[1])
                    ########
                    parts[11] = str(position[0])
                    parts[12] = str(position[1])
                    parts[13] = str(position[2])
                    file.write(' '.join(parts) + '\n')
                else:
                    file.write(line)

    @staticmethod
    def calculate_rotation_angle(position):

        x, y, z = position
        
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        unit_vector = np.array([x / magnitude, y / magnitude, z / magnitude])
        
        z_unit_vector = np.array([0, 0, 1])
        

        dot_product = np.dot(unit_vector, z_unit_vector)
        theta = np.arccos(dot_product)
        
        return theta

    def draw_bboxes(self, image_path: str, label_file_path: str, label_file_save_path: str, 
                    output_image_path: str):
        image = cv2.imread(image_path)
        imsize = image.shape[:2]
        data = self.read_data(label_file_path)
        bbox_coords_list = []
        alpha_list = []
        scale_list = []
        position_list = []
        # print(data)
        for line in data:
            label, dimension, position, rotation_y = self.parse_line(line)
            # w, l, h = dimension
            l, w, h = dimension
            scale = [h, w, l]
            x, y, z = position
            
            # perpare for the data in camera coord.
            position_cam = self.position_transform_to_camera(position)
            position_list.append([position_cam[0], position_cam[1]+h/2, position_cam[2]])
            scale_list.append(scale)
            theta = self.calculate_rotation_angle(position_cam)
            alpha = rotation_y - theta
            alpha_list.append(alpha)

 
            bbox_corners = np.array([
                [x - l / 2, y - w / 2, z - h / 2],
                [x + l / 2, y - w / 2, z - h / 2],
                [x + l / 2, y - w / 2, z + h / 2],
                [x - l / 2, y - w / 2, z + h / 2],
                [x - l / 2, y + w / 2, z - h / 2],
                [x + l / 2, y + w / 2, z - h / 2],
                [x + l / 2, y + w / 2, z + h / 2],
                [x - l / 2, y + w / 2, z + h / 2],
            ])
            # print(bbox_corners)
            print(l, w, h)

            # rotation_matrix_z = np.array([
            #     [np.cos(rotation_y), -np.sin(rotation_y), 0],
            #     [np.sin(rotation_y), np.cos(rotation_y), 0],
            #     [0, 0, 1]
            # ])
            # bbox_corners = bbox_corners.dot(rotation_matrix_z.T)
            
            bbox_corners_camera = np.zeros((8, 3))
            for i in range(bbox_corners.shape[0]):
                bbox_corners_camera[i, :] = self.transform_to_camera(bbox_corners[i], position, rotation_y)
            
            points = self.view_points(bbox_corners_camera.T, self.intrinsic_matrix)
            # points = self.view_points(bbox_corners.T, self.intrinsic_matrix)
            # print(points)
            bbox_coords = self.post_process_coords(points.T, imsize)
            bbox_coords_list.append(bbox_coords)
            
            if bbox_coords:
                min_x, min_y, max_x, max_y = bbox_coords
                # if label == "Pedestrian":
                #     color = (0, 255, 0)
                # if label == "Unknown":
                #     color = (0, 0, 255)
                color = (0, 255, 0)
                thickness = 2
                image = cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color, thickness)
        
        self.write_bbox_coord_and_alpha(label_file_save_path, data, bbox_coords_list, alpha_list, scale_list, position_list)
        cv2.imshow("Image with Rectangle", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(output_image_path, image)

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




if __name__ == "__main__":


    folder_path = r"C:\Xin Yutong\5 MA\train_data\2024_07_08-16_47_49\training"
    # folder_path = r"C:\Xin Yutong\5 MA\train_data\2024_10_09\training"
    image_folder_path = os.path.join(folder_path, "image_2")
    label_folder_save_path = os.path.join(folder_path, "label_2")
    # label_folder_path = os.path.join(folder_path, "label_2")
    label_folder_path = os.path.join(folder_path, "label_2_org")
    lidar_folder_path = os.path.join(folder_path, "lidar")
    labeled_image_folder_path = os.path.join(folder_path, "image_labeled")
    # calib_folder_path = os.path.join(folder_path, "\calib")
    calib_file_path = os.path.join(folder_path, "calib", "000000.txt")


    calib_data = read_calib_file(calib_file_path)
    intrinsic_matrix, extrinsic_matrix, R0_rect = get_calib_matrices(calib_data)

    lidar_files = os.listdir(lidar_folder_path)

    for lidar_file in lidar_files:

        file_base_name = os.path.splitext(lidar_file)[0]
        
        kitti_file_path = os.path.join(label_folder_path, f"{file_base_name}.txt")
        kitti_file_save_path = os.path.join(label_folder_save_path, f"{file_base_name}.txt")
        
        image_file_path = os.path.join(image_folder_path, f"{file_base_name}.png")
        labeled_image_file_path = os.path.join(labeled_image_folder_path, f"{file_base_name}.png")
        
        print(labeled_image_file_path)

    
        drawer = KITTIBBoxDrawer(intrinsic_matrix, extrinsic_matrix, R0_rect)
        drawer.draw_bboxes(image_file_path, kitti_file_path, kitti_file_save_path, labeled_image_file_path)
