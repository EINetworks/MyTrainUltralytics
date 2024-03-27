from ultralytics import YOLO
from test_onnx_LPRS import PlateOCR
import cv2
import os
import torch
import csv
import re

# Load a model
#for f in *.*; do ffmpeg -i "$f" -vf scale=640:640 "../lpr/${f%%.png}.png"; done
#yolo val model=runs/detect/train5/weights/best.pt data=data.yaml batch=1 imgsz=640
#yolo predict model=runs/detect/train5/weights/best.pt imgsz=640 conf=0.5 source="Indian.mp4"

model = YOLO("runs/detect/train4/weights/best.pt")  # load a pretrained model (recommended for training)




def list_jpg_files(directory):
    jpg_files = [f for f in os.listdir(directory) if f.endswith(('.Jpg','.jpg', '.png'))]
    return jpg_files

def read_from_csv(csvfile):
    jpg_files = []
    ocr_gt = []
    with open(csvfile) as file_obj:
        # Create a reader object
        reader_obj = csv.reader(file_obj)
        counter = 0
        for row in reader_obj:
            if(len(row[0]) > 0):
                jpg_files.append(row[0])
                ocr_gt.append(row[1])
    return jpg_files, ocr_gt


# Global variables to track mouse events
drawframe = None
drawing = False
ix, iy = -1, -1
bx, by = -1, -1

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, bx, by, drawing, drawframe

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw rectangle during mouse movement
            cv2.rectangle(drawframe, (ix, iy), (x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bx = x
        by = y
        # Draw final rectangle when mouse button is released
        cv2.rectangle(drawframe, (ix, iy), (x, y), (0, 255, 0), 1)



DET_W = 640
DET_H = 640

framecounter = 0
skipinitialFrames = 200
wcount = 0
font = cv2.FONT_HERSHEY_SIMPLEX


PlatesList = []

#testdir = '/mnt/ssd2/Codes/LPRSultralytics/data/ankitToll/190324_TD/images/'
#testdir = '/mnt/ssd2/DATASET/LPR/AnkitTollLPRData/160324/'
#testdir = '/mnt/ssd2/Codes/LPRSultralytics/TestImages/IND_set_10b_india_sparsh_singleline_i/'
#testdir = 'data/train/images/'

StartImageCounter = 0
#imageFileList = list_jpg_files(testdir)

CsvPath = '/mnt/ssd2/DATASET/LPR/LPR_OCR/data4/cnocrv3gtNEW.csv'
imageFileList, ocrGtList = read_from_csv(CsvPath)

#print(imageFileList)
lowScoreCount = 0

#newgttxtfile = open('cnocrv3gt.csv','w') 

for i in range(0, len(imageFileList)):
              
    imagename = imageFileList[i]
    ocrgt = ocrGtList[i]
    imagepath = imagename
    print(framecounter, ": ", imagepath, ocrgt)
   
    framecounter = framecounter + 1
    

    if framecounter < StartImageCounter:
        continue

    frame = cv2.imread(imagepath);
    
    if frame is None:
        print("Error in frame capture")
        continue
    
    origFrame = frame.copy()
    #frame = cv2.imread("multiplate2.jpg")

    lprocr, lprconf, lprminconf = PlateOCR(frame)

    print("Plate OCR: ", lprocr, " Conf: ", lprconf,", OCR MIN Conf: ", lprminconf)
      
    drawframe = frame.copy()
 
    lowScoreFound = 0
    windowName = "frame"
    k = 0
    cv2.namedWindow(windowName, 0)
    cv2.imshow(windowName, drawframe)
    if lprminconf < 0.9: #plateFound:
        lowScoreCount = lowScoreCount + 1
        lowScoreFound = 1
        k = cv2.waitKey(0)
    else:
        k = cv2.waitKey(10)


    #if lowScoreFound:
    #    outtxt = imagename + ',' + ocrgt + ',XXXXXXXXXXXXXXXXXXXX\n'
    #else:
    #    outtxt = imagename + ',' + ocrgt + ',ccccccc\n'
    #print('GT: ', outtxt)
    #newgttxtfile.write(outtxt)  

    print('lowScoreCount: ', lowScoreCount, ', framecounter: 	', framecounter)

    if k==ord('q'):
        break
    if k==ord(' '):
        cv2.waitKey(0)

#newgttxtfile.close()
        
#results = model.predict(source="./data/db2/valid/images/",  save=True, imgsz=640) # Display preds. Accepts all YOLO predict arguments , , save=True stream=True,, device="cpu"
