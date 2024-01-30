from ultralytics import YOLO
import os
import cv2
import math
import cvzone

def test_model(model):
    metrics_train = model.val(data="./data/data.yaml", plots=True, iou=0.7, split="train")
    metrics_test = model.val(data="./data/data.yaml", plots=True, iou=0.7, split="test") 

    print("\n========== Train evaluation ==========")
    print(f"Mean precision: {metrics_train.box.mp}")
    print(f"Mean recall: {metrics_train.box.mr}")
    print(f"F1 score for class 'Basketball': {metrics_train.box.f1[0]}")
    print(f"F1 score for class 'Basketball Rim': {metrics_train.box.f1[1]}")
    print(f"Mean average precision: {metrics_train.box.map50}")
    print("======================================\n")

    print("========== Test evaluation ==========")
    print(f"Mean precision: {metrics_test.box.mp}")
    print(f"Mean recall: {metrics_test.box.mr}")
    print(f"F1 score for class 'Basketball': {metrics_test.box.f1[0]}")
    print(f"F1 score for class 'Basketball Rim': {metrics_test.box.f1[1]}")
    print(f"Mean average precision: {metrics_test.box.map50}")
    print("=====================================\n")

def predict_img(image_path, output_path, model): 
    img = cv2.imread(image_path)
    results = model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = model.names[cls]
            current_class = "ball" if current_class == "Basketball" else "rim"

            if conf >= 0.3:
                if current_class == "ball":
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cvzone.putTextRect(
                    img, f'{current_class} {conf}', (x1, y1-5),
                    scale=1.5, thickness=3,
                    colorT=(255, 255, 255), colorR=color,
                    offset=3
                )
            
                cvzone.cornerRect(
                    img,
                    (x1, y1, w, h),
                    l=30, 
                    t=3,
                    rt=2,
                    colorR=color,
                    colorC=color
                )


    cv2.imwrite(output_path, img)

def predict_examples(model):
    input_folder = "./data/images"
    output_folder = "./data/predicted_images"

    os.makedirs(output_folder, exist_ok=True)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)
        predict_img(image_path, output_path, model)

if __name__ == "__main__":
    model = YOLO("./models/best.pt")

    test_model(model)
    predict_examples(model)
