import os
from ultralytics import YOLO

# Base directory for project
os.chdir("D:/Dropbox/data/dishBugs")


##### Train YOLOv9e object detector on single class #####

# Set training parameters for YOLOv9e object detector on single class"""
detection_config = {
  'model': 'yolov9e.pt',
  'data': 'D:/Dropbox/data/dishBugs/images/whole_dish/data.yaml',
  'epochs': 100,
  'imgsz': 1248,
  'patience': 50,
  'cos_lr': True,
  'single_cls': True,
  'batch': -1}  # Auto batch size

# Load YOLOv9e model
model = YOLO(detection_config['model'])

# Train detection model
results = model.train(
   data=detection_config['data'],
   epochs=detection_config['epochs'],
   imgsz=detection_config['imgsz'],
   patience=detection_config['patience'],
   cos_lr=detection_config['cos_lr'],
   single_cls=detection_config['single_cls'],
   batch=detection_config['batch'],
   verbose=True)
   
best_detection_model = model.trainer.best
print(f"Detection model training completed!")
print(f"Best weights saved to: {best_detection_model}")


##### Train YOLOv8n-cls classifier on indovidual cropped specimens #####

# Classification model configuration 
classification_config = {
    'model': 'yolov8n-cls.pt',
    'data': 'D:/Dropbox/data/dishBugs/images/isolated_obj',
    'epochs': 100,
    'imgsz': 128,
    'cos_lr': True}
        
# Load YOLOv8n classification model
model = YOLO(classification_config['model'])
        
# Train the classification model
results = model.train(
    data=classification_config['data'],
    epochs=classification_config['epochs'],
    imgsz=classification_config['imgsz'],
    cos_lr=classification_config['cos_lr'],
    verbose=True)

best_classification_model = model.trainer.best
print(f"Classification model training completed!")
print(f"Best weights saved to: {best_classification_model}")


##### Evaluate classification model on test set #####

# Load the trained model
model = YOLO(best_classification_model)

# Validate the model
results = model.val(
    data=classification_config['data'],
    split='test',
    save_txt=True,
    save_conf=True,
    plots=True,
    verbose=True)
    
print(f"Evaluation completed for model: {best_classification_model}")
