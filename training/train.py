from ultralytics import YOLO

def main():
    model = YOLO("yolo26l.pt")

    model.train(
        data="../dataset/data.yaml",
        data="../dataset/data.yaml",
        epochs=100,
        imgsz=512,
        batch=16,
        device=0,
        workers=1,
        cache=False,
        optimizer="AdamW"
    )

if __name__ == "__main__":
    main()