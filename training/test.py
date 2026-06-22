from ultralytics import YOLO
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from collections import Counter
import cv2
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from collections import Counter
import cv2
import numpy as np
import os

model = YOLO("../RoseDR_model/RoseDR.pt")
model = YOLO("../RoseDR_model/RoseDR.pt")

test_images = "dataset/test/images"
test_labels = "dataset/test/labels"

wrong_folder = "wrong_predictions"
os.makedirs(wrong_folder, exist_ok=True)

wrong_folder = "wrong_predictions"
os.makedirs(wrong_folder, exist_ok=True)

class_names = [
    "black_spot",
    "downy_mildew",
    "healthy_leaf",
    "powdery_mildew",
    "red_mite",
    "thrip"
]


def draw_wrong_image(image_path, label_path, results, save_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))

            cls = int(values[0])
            coords = values[1:]

            pts = []

            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                pts.append([x, y])

            pts = np.array(pts, dtype=np.int32)

            cv2.polylines(
                img,
                [pts],
                True,
                (0, 255, 0),
                2
            )

            x_text, y_text = pts[0]

            cv2.putText(
                img,
                f"GT:{class_names[cls]}",
                (x_text, max(20, y_text - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    if len(results[0].boxes) > 0:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                img,
                f"Pred:{class_names[int(cls)]} {conf:.2f}",
                (x1, min(h - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    cv2.imwrite(save_path, img)



def draw_wrong_image(image_path, label_path, results, save_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))

            cls = int(values[0])
            coords = values[1:]

            pts = []

            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                pts.append([x, y])

            pts = np.array(pts, dtype=np.int32)

            cv2.polylines(
                img,
                [pts],
                True,
                (0, 255, 0),
                2
            )

            x_text, y_text = pts[0]

            cv2.putText(
                img,
                f"GT:{class_names[cls]}",
                (x_text, max(20, y_text - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    if len(results[0].boxes) > 0:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                img,
                f"Pred:{class_names[int(cls)]} {conf:.2f}",
                (x1, min(h - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    cv2.imwrite(save_path, img)


y_true = []
y_pred = []

wrong_count = 0

wrong_count = 0

for image_file in os.listdir(test_images):

    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(test_images, image_file)

    label_path = os.path.join(
    label_path = os.path.join(
        test_labels,
        os.path.splitext(image_file)[0] + ".txt"
    )

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        gt_classes = [
            int(line.split()[0])
            for line in f
            if line.strip()
        ]

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        gt_classes = [
            int(line.split()[0])
            for line in f
            if line.strip()
        ]

    results = model(image_path, verbose=False)

    pred_classes = []

    if len(results[0].boxes) > 0:
        pred_classes = (
            results[0]
            .boxes
            .cls
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

    if Counter(gt_classes) != Counter(pred_classes):

        draw_wrong_image(
            image_path,
            label_path,
            results,
            os.path.join(wrong_folder, image_file)
        )

        wrong_count += 1

    max_len = max(len(gt_classes), len(pred_classes))

    gt_extended = gt_classes + [-1] * (max_len - len(gt_classes))
    pred_extended = pred_classes + [-1] * (max_len - len(pred_classes))

    y_true.extend(gt_extended)
    y_pred.extend(pred_extended)

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )
)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )
)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print(f"\nWrong images: {wrong_count}")
print(f"Saved to: {wrong_folder}")

metrics = model.val(
    data="dataset/data.yaml",
    split="test",
    workers=0,
    verbose=False
)

print("\nDetection Metrics:\n")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")

det_f1 = (
    2 * metrics.box.mp * metrics.box.mr
    / (metrics.box.mp + metrics.box.mr + 1e-16)
)

print(f"F1 Score:  {det_f1:.4f}")