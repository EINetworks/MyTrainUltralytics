from ultralytics import YOLO
from test_onnx_LPRS import PlateOCR
import cv2

# Load a model
#for f in *.*; do ffmpeg -i "$f" -vf scale=640:640 "../lpr/${f%%.png}.png"; done
#yolo val model=runs/detect/train5/weights/best.pt data=data.yaml batch=1 imgsz=640
#yolo predict model=runs/detect/train5/weights/best.pt imgsz=640 conf=0.5 source="Indian.mp4"

model = YOLO("runs/detect/train5/weights/best.pt")  # load a pretrained model (recommended for training)

#vid = "/mnt/ssd2/DATASET/LPR/LPRSPramaCamera/cc4.mp4"
#vid = 'Indian.mp4' #'AnkitToll.mp4'
vid = 'rtsp://admin:Rst12345@122.160.49.247:5002/Streaming/Channels/101/'
cap = cv2.VideoCapture(vid)

DET_W = 640
DET_H = 640

skip = 5
framecounter = 0
skipinitialFrames = 0
wcount = 0
font = cv2.FONT_HERSHEY_SIMPLEX


PlatesList = []
while 1:
    ret, frame = cap.read()
    if not ret:
        print("Error in frame capture")
        break
    #frame = cv2.imread('test.jpg')
    if frame is None:
        print("Error in frame capture")
        break
    framecounter = framecounter + 1
    if framecounter%skip != 0:
        continue

    if framecounter<skipinitialFrames: 
        continue;

    #frame = cv2.imread("multiplate2.jpg")

    frameScaled = cv2.resize(frame, (DET_W,DET_H))
    results = model.predict(frameScaled, device = 'cpu', show=False)
    print("length is ", len(results))

    scalex = frame.shape[1] / float(DET_W)
    scaley = frame.shape[0] / float(DET_H)
    print('scalex: ', scalex, ', scaley: ',scaley)

    plateFound = 0
    PlateOutputList = []
    for result in results:
        boxes = result.boxes.numpy()  # Boxes object for bounding box outputs
        #masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs
        #print(' boxes.xyxy: ',  boxes.xyxy, len(boxes.xyxy))

        print(boxes.shape)
        if(len(boxes.xyxy) == 0) :
            continue
        for i in range(0, boxes.shape[0]):
            print('Det conf: ', boxes.conf[i])
            x1 = boxes.xyxy[i][0]*scalex
            y1 = boxes.xyxy[i][1]*scaley
            x2 = boxes.xyxy[i][2]*scalex
            y2 = boxes.xyxy[i][3]*scaley

            #print("box is",x1,y1,x2,y2)
            plateimage = frame[(int)(y1):(int)(y2), (int)(x1): (int)(x2)]
            print("Detected Plate size: ", plateimage.shape)

            dispname = "plateimage_" + str(i)
            cv2.imshow(dispname, plateimage)
            lprocr, lprconf = PlateOCR(plateimage)
            print("Plate OCR: ", lprocr, " Conf: ", lprconf)
            
            PlateOutputList.append((x1,y1,x2,y2,boxes.conf[i],lprocr, lprconf))
            plateFound = 1
        #result.show()  # display to screen

    #print(PlateOutputList)
    for cureplate in PlateOutputList:
        x1,y1,x2,y2,detconf,lprocr,ocrconf = cureplate
        if ocrconf > 0.8:
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 5)
            myh = (int)(y2-y1) + 40
            cv2.putText(frame, lprocr, ((int)(x1), (int)(y1)+myh), font, 1.0, (0,0,255) , thickness=3, lineType=cv2.LINE_AA)
           
            if lprocr not in PlatesList:
                PlatesList.append(lprocr)
        else:
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 5)

    PlateListLast = PlatesList[-5:]
    dispcnt = 0
    for platetxt in PlateListLast:
        cv2.putText(frame, platetxt, (50, 50*dispcnt + 50), font, 1.0, (0,0,255) , thickness=3, lineType=cv2.LINE_AA)
        dispcnt = dispcnt + 1

    cv2.namedWindow("frame", 0)
    cv2.imshow("frame", frame)
    if plateFound:
        k = cv2.waitKey(10)
    else:
        k = cv2.waitKey(10)

    if k==ord('q'):
        break
    if k==ord(' '):
        cv2.waitKey(0)

#results = model.predict(source="./data/db2/valid/images/",  save=True, imgsz=640) # Display preds. Accepts all YOLO predict arguments , , save=True stream=True,, device="cpu"
