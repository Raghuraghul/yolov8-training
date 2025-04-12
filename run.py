#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path

if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator

def main():
    source = 4
    output_path = "output.mp4"
    yolo_model_size = "nano"
    depth_model_size = "small"
    device = 'cpu'
    conf_threshold = 0.25
    iou_threshold = 0.45
    classes = None  # Add 'pallet' class if custom trained
    enable_tracking = True
    enable_bev = True
    enable_pseudo_3d = True
    camera_params_file = None
    
    print(f"Using device: {device}")
    
    print("Initializing models...")
    detector = ObjectDetector(
        model_size=yolo_model_size,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        classes=classes,
        device=device
    )
    depth_estimator = DepthEstimator(
        model_size=depth_model_size,
        device=device
    )
    bbox3d_estimator = BBox3DEstimator()
    if enable_bev:
        bev = BirdEyeView(scale=60, size=(300, 300))
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    print("Starting processing...")
    while True:
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            print("Exiting program...")
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        detection_frame = frame.copy()
        depth_frame = frame.copy()
        result_frame = frame.copy()
        
        # Object Detection with Tracking
        detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
        
        # Depth Estimation
        depth_map = depth_estimator.estimate_depth(original_frame)
        depth_colored = depth_estimator.colorize_depth(depth_map)
        
        # 3D Bounding Box Estimation
        boxes_3d = []
        active_ids = []
        for detection in detections:
            bbox, score, class_id, obj_id = detection
            class_name = detector.get_class_names()[class_id]
            
            if 'pallet' in class_name.lower():
                depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='custom')
            else:
                depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
            
            box_3d = bbox3d_estimator.estimate_3d_box(bbox, depth_value, class_name, obj_id)
            box_3d['score'] = score
            boxes_3d.append(box_3d)
            if obj_id is not None:
                active_ids.append(obj_id)
        
        bbox3d_estimator.cleanup_trackers(active_ids)
        
        # Visualization
        for box_3d in boxes_3d:
            class_name = box_3d['class_name'].lower()
            color = (255, 0, 255) if 'pallet' in class_name else \
                    (0, 0, 255) if 'car' in class_name else \
                    (0, 255, 0) if 'person' in class_name else (255, 255, 255)
            result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
        
        if enable_bev:
            bev.reset()
            for box_3d in boxes_3d:
                bev.draw_box(box_3d)
            bev_image = bev.get_image()
            bev_height = height // 4
            bev_width = bev_height
            bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
            result_frame[height - bev_height:height, 0:bev_width] = bev_resized
            cv2.rectangle(result_frame, (0, height - bev_height), (bev_width, height), (255, 255, 255), 1)
            cv2.putText(result_frame, "Bird's Eye View", (10, height - bev_height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS Calculation
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_value = frame_count / elapsed_time
            fps_display = f"FPS: {fps_value:.1f}"
        
        cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add depth map
        depth_height = height // 4
        depth_width = depth_height * width // height
        depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
        result_frame[0:depth_height, 0:depth_width] = depth_resized
        
        out.write(result_frame)
        cv2.imshow("3D Object Detection", result_frame)
        cv2.imshow("Depth Map", depth_colored)
        cv2.imshow("Object Detection", detection_frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()