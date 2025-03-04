"""
This is an inference snippet code follow YOLO official doc format.
"""

from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

results = model(["<SET YOUR IMAGES HERE TO DO DEMONSTRATION>"])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen