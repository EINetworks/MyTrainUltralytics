
#env activate
source yoloultravenv/bin/activate

#training
python train.py

#testing
yolo predict model=yolov8n.pt show=True imgsz=640 device="cpu" conf=0.25 source="AnkitToll.mp4"

yolo predict model=runs/detect/train5/weights/best.pt show=True device="cpu" conf=0.25 source="AnkitToll.mp4" imgsz=1280
yolo predict model=runs/detect/train5/weights/best.pt show=True conf=0.25 source="AnkitToll.mp4" imgsz=1280

#transfer to mercury

scp -P2224 runs/detect/train8/weights/best.onnx mercury@122.160.74.98:/tmp/lp2.onnx

# to visualize
python vis_bbox.py --folder_path data/train/images_labels/ --classes temp.classes --save_path testout/
