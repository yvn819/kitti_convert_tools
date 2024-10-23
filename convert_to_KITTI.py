import numpy as np
import json
import os
from read_json import ReadJson

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
    # P0 = calib_data['K'].reshape(3, 4)
    # Tr_velo_to_cam = calib_data['T'].reshape(3, 4)
    # R0_rect = calib_data['R0_rect'].reshape(3, 3)
    return P0, Tr_velo_to_cam, R0_rect

# intrinsic_matrix = np.array([[969.94239058, 0, 956.28949532, 0],
#                              [0, 971.9671513, 556.70141901, 0],
#                              [0, 0, 1, 0]], dtype=float)

# extrinsic_matrix = np.array([[-0.0114814, -0.999882, 0.0101998, 0.23438],
#                             [0.0421775, -0.0106757, -0.999053, -0.452339],
#                             [0.999045, -0.0110404, 0.0422951, -0.543866]], dtype=float)

# intrinsic_matrix = np.array([[1040.666417, 0.000000, 786.492785, 0],
#                              [0.000000, 1131.537713, 404.782049, 0],
#                              [0, 0, 1, 0]], dtype=float)

# extrinsic_matrix = np.array([[0.0237887, -0.999715, 0.00210834, 0.132865],
#                             [0.0989612, 0.000256231, -0.995091, -0.423957],
#                             [0.994807, 0.0238806, 0.098939, -0.703353]], dtype=float)

# folder_path = r"G:\MA\labeled_data\2024_07_08-16_40_41\training\label_json"
# output_folder_path = r"G:\MA\labeled_data\2024_07_08-16_40_41\training\label_2_org"


_folder_path = r"C:\Xin Yutong\5 MA\train_data\2024_07_08-16_47_49\training"
folder_path = os.path.join(_folder_path, "label_json")
output_folder_path = os.path.join(_folder_path, "label_2_org")
calib_file_path = os.path.join(_folder_path, "calib", "000000.txt")

calib_data = read_calib_file(calib_file_path)
intrinsic_matrix, extrinsic_matrix, R0_rect = get_calib_matrices(calib_data)

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        
        reader = ReadJson(file_path, intrinsic_matrix, extrinsic_matrix)

        reader.load_json_files_from_folder()
        labels = reader.extract_info()

        # print(labels)

        output_file = filename[:-5] + '.txt'
        output_file_path = os.path.join(output_folder_path, output_file)


        if labels:
            # print(label)
            reader.save_data_to_txt(labels, output_file_path)

