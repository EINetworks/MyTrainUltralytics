from ultralytics import YOLO
import cv2

# Load a model
#for f in *.*; do ffmpeg -i "$f" -vf scale=640:640 "../lpr/${f%%.png}.png"; done
#yolo val model=runs/detect/train5/weights/best.pt data=data.yaml batch=1 imgsz=640
#yolo predict model=runs/detect/train5/weights/best.pt imgsz=640 conf=0.5 source="Indian.mp4"

model = YOLO("runs/detect/train5/weights/best.pt")  # load a pretrained model (recommended for training)

results = model.predict(source="AnkitToll.mp4",  save=True, show=True, imgsz=640) # Display preds. Accepts all YOLO predict arguments , , save=True stream=True,, device="cpu"
