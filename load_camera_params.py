import os
import json
import numpy as np
from pathlib import Path

def load_camera_params(params_file):
    if not os.path.exists(params_file):
        print(f"Warning: Camera parameters file {params_file} not found. Using defaults.")
        return None
    
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        params['camera_matrix'] = np.array(params['camera_matrix'])
        params['dist_coeffs'] = np.array(params['dist_coeffs'])
        params['projection_matrix'] = np.array(params['projection_matrix'])
        print(f"Loaded camera parameters from {params_file}")
        return params
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        return None

def create_projection_matrix(camera_matrix, R=None, t=None):
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros((3, 1))
    RT = np.hstack((R, t))
    return camera_matrix @ RT

def apply_camera_params_to_estimator(bbox3d_estimator, params):
    if params is None:
        print("Warning: No camera parameters provided. Using defaults.")
        return bbox3d_estimator
    if 'camera_matrix' in params:
        bbox3d_estimator.K = params['camera_matrix']
    if 'projection_matrix' in params:
        bbox3d_estimator.P = params['projection_matrix']
    print("Applied camera parameters to 3D bounding box estimator")
    return bbox3d_estimator

def main():
    params_file = "camera_params.json"
    params = load_camera_params(params_file)
    if params:
        print(f"Image dimensions: {params['image_width']}x{params['image_height']}")
        print(f"Reprojection error: {params['reprojection_error']}")
        R = np.eye(3)
        t = np.array([[0], [1.65], [0]])
        projection_matrix = create_projection_matrix(params['camera_matrix'], R, t)
        print(f"New projection matrix:\n{projection_matrix}")

if __name__ == "__main__":
    main()