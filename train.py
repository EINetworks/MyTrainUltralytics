from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8s.yaml")  # build a new model from scratch
#model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

#finetune
#model = YOLO("runs/detect/train2/weights/best.pt")
model = YOLO("runs/detect/train8/weights/best.pt")


# Use the model
model.train(data="data.yaml", epochs=10, batch=32)#, lr0 = 0.01, lrf=0.0001)#, resume = True)  # train the model


metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
