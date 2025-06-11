import os
from ultralytics import YOLO

# Base directory for project
os.chdir("D:/Dropbox/data/dishBugs")

# Path to pretrained object detection model
model_path='D:/Dropbox/data/dishBugs/runs/detect/train/weights/best.pt' 

# Path to images to run inference on
source_path='D:/Dropbox/data/dishBugs/images/whole_dish/valid/images'


##### Run object detection and save cropped detections #####

# Load the detection model
model = YOLO(model_path)

# Run prediction with save_crop=True to save cropped detections
results = model.predict(
    source=source_path,
    save_crop=True,
    verbose=True)
    
print(f"Object detection completed. Crops saved to prediction folder.")


##### Run classification on cropped detections #####

# Path to pretrained object detection model
model_path='D:/Dropbox/data/dishBugs/runs/classify/train/weights/best.pt'

# Path to cropped bounding boxes to run inference on
crops_path='D:/Dropbox/data/dishBugs/images/isolated_obj/val'

# Load classification model
model = YOLO(model_path)

# Run validation/prediction on crops
results = model.predict(
  source=str(crops_path),
  save=True,
  save_txt=True,
  save_conf=True,
  verbose=True)
  
print(f"Classification completed on cropped detections.")

if results:
  print("\nClassification Results Summary:")
  for i, result in enumerate(results):
    if hasattr(result, 'probs') and result.probs is not None:
      top1_idx = result.probs.top1
      confidence = result.probs.top1conf.item()
      if hasattr(result.probs, 'names'):
        class_name = result.probs.names[top1_idx]
        print(f"Image {i+1}: {class_name} (confidence: {confidence:.3f})")
      else:
        print(f"Image {i+1}: Class {top1_idx} (confidence: {confidence:.3f})")
