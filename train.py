from ultralytics import YOLO

model = YOLO("yolo12n.pt")

dataset_path = "<PLEASE SPECIFY ME>"

results = model.train(
    data=f"{dataset_path}/data.yaml", 
    epochs=60, 
    device=0 
)