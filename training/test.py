from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import os

model = YOLO("runs/detect/train/weights/best.pt")

test_images = "dataset/test/images"
test_labels = "dataset/test/labels"

class_names = [
    "black_spot",
    "downy_mildew",
    "healthy_leaf",
    "powdery_mildew",
    "red_mite",
    "thrip"
]

y_true = []
y_pred = []

for image_file in os.listdir(test_images):

    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(test_images, image_file)

    label_file = os.path.join(
        test_labels,
        os.path.splitext(image_file)[0] + ".txt"
    )

    with open(label_file, "r") as f:
        first_line = f.readline().strip()

    gt_class = int(first_line.split()[0])

    # Prediction
    results = model(image_path, verbose=False)

    if len(results[0].boxes) > 0:
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        best_idx = confs.argmax()
        pred_class = int(classes[best_idx])
    else:
        pred_class = -1

    y_true.append(gt_class)
    y_pred.append(pred_class)

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))