# train_yolov8.py

from ultralytics import YOLO

# Load the YOLOv8 model
# Use a pre-trained model or your custom model
# For example, 'yolov8n.pt' is the Nano model; choose others as needed
model = YOLO("yolov8n.pt")  # Update path if using a custom model

# Train the model
model.train(
    data="data.yaml",               # Path to your updated data.yaml
    epochs=100,                     # Number of training epochs (adjust as needed)
    imgsz=640,                      # Image size for training
    batch=16,                       # Batch size (adjust based on GPU memory)
    name="biodegradable_detection", # Name for the training run
    cache=True                      # Cache images for faster training
)
