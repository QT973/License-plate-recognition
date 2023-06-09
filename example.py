from ultralytics import YOLO
# Load a pretrained model
model = YOLO("yolov8n.pt")

if __name__ == "__main__":
    # use the model
    model.train(data="coco128.yaml", epochs=100, imgsz=640,
                device="0", batch=20, workers=2)  # train the model
