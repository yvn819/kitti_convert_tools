import json
import os
import numpy as np


class ReadJson:
    def __init__(self, file_path, intrinsic_matrix, extrinsic_matrix):
        self.file_path = file_path
        self.json_data = []
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix


    def load_json_files_from_folder(self):
        """Load all JSON files from a specified folder and return their content."""
        
        # Iterate over all files in the specified folder

        with open(self.file_path, 'r') as file:
            data = json.load(file)
            self.json_data.append(data)
        
        return self.json_data
    
    def extract_info(self):
        """Extract relevant barrier information from JSON data."""
        labels = []
        
        # Iterate through each item in the json_data
        for items in self.json_data:

            for item in items:

                label_info = {
                    'id': item['obj_id'],
                    'category': item.get('obj_type'),
                    'position': {
                        'x': item.get('psr', {}).get('position', {}).get('x', 0),
                        'y': item.get('psr', {}).get('position', {}).get('y', 0),
                        'z': item.get('psr', {}).get('position', {}).get('z', 0),
                    },
                    'rotation': {
                        'x': item.get('psr', {}).get('rotation', {}).get('x', 0),
                        'y': item.get('psr', {}).get('rotation', {}).get('y', 0),
                        'z': item.get('psr', {}).get('rotation', {}).get('z', 0),
                    },
                    'scale': {
                        'x': item.get('psr', {}).get('scale', {}).get('x', 0),
                        'y': item.get('psr', {}).get('scale', {}).get('y', 0),
                        'z': item.get('psr', {}).get('scale', {}).get('z', 0),
                    }
                }
                # Append the dictionary to the list
                labels.append(label_info)

        
        return labels
    
    def transform_to_camera(self, location):
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
    
    def save_data_to_txt(self, labels, output_file_path):
        """Save data to a text file with each ID on a new line and fields separated by spaces."""

        with open(output_file_path, 'w') as file:
            for label in labels:
                category = label['category']
                is_truncated = 0
                is_occluded = 0
                alpha = 0
                bbox = [0, 0, 0, 0]
                x = label['scale']['x']
                y = label['scale']['y']
                z = label['scale']['z']

                location = [label['position']['x'], label['position']['y'], label['position']['z']]
                # location_cam = self.transform_to_camera(location)
                location_cam = location
                # print(location_cam)
                roation_y = label['rotation']['z']

                line = (f"{category} {is_truncated} {is_occluded} {alpha} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} "
                        f"{x} {y} {z} {location_cam[0]} {location_cam[1]} {location_cam[2]} {roation_y} \n")
                    
                file.write(line)