from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("./models/yolov8n.pt")

    results = model.train(data='./data/data.yaml', epochs=100, imgsz=640, optimizer="SGD", lr0=0.01, patience=50, batch=8, val=True)

    # results = model.train(data='./data/data.yaml', epochs=100, imgsz=640, optimizer="Adam", lr0=0.001, patience=50, batch=8, val=True)