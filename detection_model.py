import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque

class ObjectDetector:
    def __init__(self, model_size='small', conf_thres=0.25, iou_thres=0.45, classes=None, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        if self.device == 'mps':
            print("Using MPS device with CPU fallback")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        print(f"Using device: {self.device} for object detection")
        
        model_map = {
            'nano': 'yolo11n',
            'small': 'yolo11s',
            'medium': 'yolo11m',
            'large': 'yolo11l',
            'extra': 'yolo11x'
        }
        
        model_name = '/home/raghul/Downloads/yolo_training/runs/detect/train5/weights/best.pt'
        
        try:
            self.model = YOLO(model_name)
            print(f"Loaded YOLOv11 {model_size} model on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = YOLO(model_name)
        
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        
        if classes is not None:
            self.model.overrides['classes'] = classes
        
        # Custom tracking variables
        self.tracked_objects = {}  # {id: [centroid, bbox, class_id]}
        self.next_id = 0
        self.max_distance = 50  # Max distance for matching centroids
        self.max_lost = 5  # Frames to keep lost objects
        self.lost_objects = {}  # {id: [frames_lost, last_bbox, class_id]}

    def detect(self, image, track=True):
        detections = []
        annotated_image = image.copy()
        
        try:
            results = self.model.predict(image, verbose=False, device=self.device)
        except RuntimeError as e:
            if self.device == 'mps' and "not currently implemented" in str(e):
                print(f"MPS error: {e}, falling back to CPU")
                results = self.model.predict(image, verbose=False, device='cpu')
            else:
                raise
        
        # Process detections
        for predictions in results:
            if predictions is None or predictions.boxes is None:
                continue
            
            boxes = predictions.boxes.xyxy.cpu().numpy()
            scores = predictions.boxes.conf.cpu().numpy()
            classes = predictions.boxes.cls.cpu().numpy()
            
            current_centroids = []
            for i, (bbox, score, cls) in enumerate(zip(boxes, scores, classes)):
                xmin, ymin, xmax, ymax = bbox
                centroid = ((xmin + xmax) / 2, (ymin + ymax) / 2)
                current_centroids.append((centroid, bbox, int(cls), float(score)))
                
                # Draw detection (without ID yet)
                cv2.rectangle(annotated_image, (int(xmin), int(ymin)), 
                            (int(xmax), int(ymax)), (0, 0, 225), 2)
                label = f"{predictions.names[int(cls)]} {score:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                dim, baseline = text_size[0], text_size[1]
                cv2.rectangle(annotated_image, (int(xmin), int(ymin)), 
                            (int(xmin) + dim[0], int(ymin) - dim[1] - baseline), 
                            (30, 30, 30), cv2.FILLED)
                cv2.putText(annotated_image, label, (int(xmin), int(ymin) - 7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if track:
                # Update tracking
                detections = self._update_tracking(current_centroids, annotated_image)
            else:
                for centroid, bbox, cls, score in current_centroids:
                    detections.append([bbox.tolist(), score, cls, None])
        
        return annotated_image, detections

    def _update_tracking(self, current_centroids, image):
        detections = []
        
        # Match current centroids with tracked objects
        used_ids = set()
        new_tracked = {}
        
        for centroid, bbox, cls, score in current_centroids:
            best_id = None
            min_dist = float('inf')
            
            for obj_id, (old_centroid, old_bbox, old_cls) in self.tracked_objects.items():
                if obj_id in used_ids or old_cls != cls:
                    continue
                dist = np.sqrt((centroid[0] - old_centroid[0])**2 + 
                             (centroid[1] - old_centroid[1])**2)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_id = obj_id
            
            if best_id is not None:
                new_tracked[best_id] = (centroid, bbox, cls)
                used_ids.add(best_id)
            else:
                # New object
                new_id = self.next_id
                self.next_id += 1
                new_tracked[new_id] = (centroid, bbox, cls)
                used_ids.add(new_id)
            
            detections.append([bbox.tolist(), score, cls, best_id if best_id is not None else new_id])

        # Update tracked objects
        self.tracked_objects = new_tracked

        # Handle lost objects
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in used_ids:
                if obj_id not in self.lost_objects:
                    self.lost_objects[obj_id] = [0, self.tracked_objects[obj_id][1], self.tracked_objects[obj_id][2]]
                self.lost_objects[obj_id][0] += 1
                if self.lost_objects[obj_id][0] > self.max_lost:
                    del self.lost_objects[obj_id]
                    del self.tracked_objects[obj_id]

        # Draw tracked objects with IDs
        for obj_id, (centroid, bbox, cls) in self.tracked_objects.items():
            xmin, ymin, xmax, ymax = bbox
            label = f"ID: {obj_id} {self.model.names[cls]}"
            cv2.putText(image, label, (int(xmin), int(ymin) - 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(image, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)

        return detections

    def get_class_names(self):
        return self.model.names