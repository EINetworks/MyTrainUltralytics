from ultralytics import YOLO
from test_onnx_LPRS import PlateOCR
import cv2
import os
import torch
import csv

# Load a model
#for f in *.*; do ffmpeg -i "$f" -vf scale=640:640 "../lpr/${f%%.png}.png"; done
#yolo val model=runs/detect/train5/weights/best.pt data=data.yaml batch=1 imgsz=640
#yolo predict model=runs/detect/train5/weights/best.pt imgsz=640 conf=0.5 source="Indian.mp4"

model = YOLO("runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)




def list_jpg_files(directory):
    jpg_files = [f for f in os.listdir(directory) if f.endswith(('.Jpg','.jpg'))]
    return jpg_files

def read_from_csv(csvfile):
    jpg_files = []
    with open(csvfile) as file_obj:
        # Create a reader object
        reader_obj = csv.reader(file_obj)
        counter = 0
        for row in reader_obj:
            if(len(row[1]) > 0):
                if row[20] == 'Bus' or row[20] == 'Truck' or row[17] == 'Bus' or row[17] == 'Truck' or 'Axle' in row[17] or 'Axle' in row[20]:
                    print(counter, ': ', row[1], ', 17: ', row[17], ', 20: ', row[20])
                    imgname = row[1] + '.Jpg'
                    jpg_files.append(imgname)
                    imgname = row[1] + 'F.Jpg'
                    jpg_files.append(imgname)
    return jpg_files


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

testdir = '/mnt/ssd2/Codes/LPRSultralytics/data/ankitToll/190324_TD/images/'
#testdir = '/mnt/ssd2/DATASET/LPR/AnkitTollLPRData/160324/'
#testdir = 'data/train/images/'
CsvPath= '/mnt/ssd2/DATASET/LPR/AnkitTollLPRData/16March24.csv'
#'data/ankitToll/230124_AllGT/230124_GT/images/'

GToutpath = '/mnt/ssd2/DATASET/LPR/AnkitTollLPRData/160324_TD/'
StartImageCounter = 0
writeGT = 0

DeleteLowconfPlates = 0
DeletionCount = 0

imageFileList = list_jpg_files(testdir)
#imageFileList = read_from_csv(CsvPath)

#print(imageFileList)
for imagename in imageFileList:
              
    #if imagename.lower().startswith(('2403190202', '2403190203', '2403190208')) == 0 or imagename.lower().endswith('f.jpg') == 1:
    #    continue
    #if imagename.lower().startswith(('2401260310')) == 0 : 
    #    continue

    imagepath = testdir + imagename
    print(framecounter, ": ", imagepath)
   
    framecounter = framecounter + 1

    if framecounter < StartImageCounter:
        continue

    frame = cv2.imread(imagepath);

    if frame is None:
        print("Error in frame capture")
        continue
    
    origFrame = frame.copy()
    #frame = cv2.imread("multiplate2.jpg")

    imgH, imgW, _ = frame.shape
    scale = min(float(DET_W)/imgW, float(DET_H)/imgH)
    scaledW = (int)(imgW*scale)
    scaledH = (int)(imgH*scale)
    scaledW32 = scaledW
    if scaledW%32 != 0:
        scaledW32 = scaledW + (32 - scaledW%32)
    scaledH32 = scaledH
    if scaledH%32 != 0:
        scaledH32 = scaledH + (32 - scaledH%32)
    print(scaledW, scaledH, scaledW32, scaledH32)

    frameScaled = cv2.resize(frame, (scaledW,scaledH))
    borderx = (scaledW32-scaledW)
    bordery = (scaledH32-scaledH)
    frameScaled = cv2.copyMakeBorder(frameScaled, 0, (int)(bordery), 0, (int)(borderx), cv2.BORDER_REPLICATE)
    cv2.imshow("frameScaled", frameScaled)

    frameScaledTorch = torch.from_numpy(frameScaled).permute(2, 0, 1).unsqueeze_(0)
    frameScaledTorch = frameScaledTorch.to(dtype=torch.float, device='cuda')
    frameScaledTorch.div_(255.)
    #frameScaledTorch = torch.from_numpy(frameScaled.transpose(2,0,1))
    #frameScaledTorch = frameScaledTorch.unsqueeze(0)/255.0
    #results = model.predict(frameScaledTorch, device = 'cuda', conf = 0.25, show=False)	## Running on GPU has some accuracy issues in conversion to torch
    results = model.predict(frameScaled, device = 'cpu', conf = 0.25, show=False)	#source='/tmp/240319020200123.Jpg'

    #print("length is ", len(results))

    scalex = frame.shape[1] / float(scaledW)
    scaley = frame.shape[0] / float(scaledH)
    #print('scalex: ', scalex, ', scaley: ',scaley)

    plateFound = 0
    lowconfplatefound = 0
    highconfplatefound = 0
    PlateOutputList = []
    minOcrConf = 1.0
    for result in results:
        boxes = result.boxes.cpu().numpy() # Boxes object for bounding box outputs
        #masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs
        #print(' boxes.xyxy: ',  boxes.xyxy, len(boxes.xyxy))

        print('boxes.shape: ', boxes.shape)
        if(len(boxes.xyxy) == 0) :
            continue
        for i in range(0, boxes.shape[0]):
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>.  Detection  conf: ', boxes.conf[i])
            x1 = boxes.xyxy[i][0]*scalex
            y1 = boxes.xyxy[i][1]*scaley
            x2 = boxes.xyxy[i][2]*scalex
            y2 = boxes.xyxy[i][3]*scaley

            #print("box is",x1,y1,x2,y2)
            plateimage = frame[(int)(y1):(int)(y2), (int)(x1): (int)(x2)]
            print("Detected Plate size: ", plateimage.shape)
            #if boxes.conf[i] < 0.5:
            #    lowconfplatefound = 1

            #dispname = "plateimage_" + str(i)
            #cv2.imshow(dispname, plateimage)
            lprocr, lprconf = PlateOCR(plateimage)
            print("Plate OCR: ", lprocr, " OCR Conf: ", lprconf, ", Detection Conf: ", boxes.conf[i])
            minOcrConf = min(minOcrConf, lprconf)
            
            PlateOutputList.append((x1,y1,x2,y2,boxes.conf[i],lprocr, lprconf))
            plateFound = 1
        #result.show()  # display to screen

    #print(PlateOutputList)
    for cureplate in PlateOutputList:
        x1,y1,x2,y2,detconf,lprocr,ocrconf = cureplate
        if ocrconf > 0.8 or detconf > 0.4:
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 5)
            myh = (int)(y2-y1) + 40
            cv2.putText(frame, lprocr, ((int)(x1), (int)(y1)+myh), font, 1.0, (0,0,255) , thickness=3, lineType=cv2.LINE_AA)
           
            if lprocr not in PlatesList:
                PlatesList.append(lprocr)
            highconfplatefound = 1
        else:
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 5)
            if x2-x1 > 50 and y2-y1 > 20:
                lowconfplatefound = 1

    PlateListLast = PlatesList[-5:]
    dispcnt = 0
    #for platetxt in PlateListLast:
    #    cv2.putText(frame, platetxt, (50, 50*dispcnt + 50), font, 1.0, (0,0,255) , thickness=3, lineType=cv2.LINE_AA)
    #    dispcnt = dispcnt + 1

    drawframe = frame.copy()
    if writeGT:
         cv2.putText(drawframe, "Press 'f' for FA, 'a' for TD, 'm' for manual GT",  (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    windowName = "frame"
    cv2.namedWindow(windowName, 0)
    cv2.imshow(windowName, drawframe)
    if lowconfplatefound == 1 and highconfplatefound == 0: #plateFound:
        k = cv2.waitKey(0)
    else:
        k = cv2.waitKey(0)


    if k==ord('q'):
        break
    if k==ord(' '):
        cv2.waitKey(0)

    ### save image as negative
    if k == ord('f') and writeGT==1:
        lastpos = imagename.rfind('.')
        outgtfile = GToutpath + imagename[0:lastpos] + '.txt'
        outfilename = GToutpath + imagename
        print("FA >>>>>>>>>>>>>>", imagename, outfilename)
     
        cv2.imwrite(outfilename, origFrame)
        gttxtfile = open(outgtfile,'w') 
        gttxtfile.close()
        #outtxt = outfilename + '\t' + outjsonstr + '\n'
        print(outgtfile)

    ### save image as TD
    if k == ord('a') and writeGT==1:
        lastpos = imagename.rfind('.')
        outgtfile = GToutpath + imagename[0:lastpos] + '.txt'
        outfilename = GToutpath + imagename
        print("TD >>>>>>>>>>>>>>", imagename, outfilename, )
     
        cv2.imwrite(outfilename, origFrame)
        gttxtfile = open(outgtfile,'w') 
        for cureplate in PlateOutputList:
            x1,y1,x2,y2,detconf,lprocr,ocrconf = cureplate
            imgh, imgw, _ = frame.shape
            platecx = ((x1+x2)/2) / imgw
            platecy = ((y1+y2)/2) / imgh
            pw = (x2-x1) / imgw
            ph = (y2-y1) / imgh
            outtxt = '0 ' + str(round(platecx,5)) + ' ' + str(round(platecy,5)) + ' ' + str(round(pw,5)) + ' ' + str(round(ph,5))
            print('GT: ', outtxt)
            gttxtfile.write(outtxt)  
        gttxtfile.close()
        #outtxt = outfilename + '\t' + outjsonstr + '\n'
        print(outgtfile)


    ### save image as TD
    if k == ord('m') and writeGT==1:
        drawframe = frame.copy()
        
        cv2.setMouseCallback(windowName, draw_rectangle)
        cv2.putText(drawframe, "Mark ROI by mouse. Press Esc to continue",  (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        ManualPlateList = []
        while True:
            
            cv2.imshow(windowName, drawframe)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                print("Marked Rectangle: ", ix, iy, bx, by)
                drawframe = frame.copy()
                cv2.rectangle(drawframe, (ix, iy), (bx, by), (0,255,255), 4) 
                cv2.putText(drawframe, "Press a to accept, any other key to reject", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.imshow(windowName, drawframe)
                k = cv2.waitKey(0)
                print("Pressed key is: ", k)
                if k==97:
                    ManualPlateList.append((ix, iy, bx, by, ""))
                    print("New GT markup done")
                    ### save to file
                break
        lastpos = imagename.rfind('.')
        outgtfile = GToutpath + imagename[0:lastpos] + '.txt'
        outfilename = GToutpath + imagename
        print(">>>>>>>>>>>>>>", imagename, outfilename)
     
        cv2.imwrite(outfilename, origFrame)
        gttxtfile = open(outgtfile,'w') 
        for cureplate in ManualPlateList:
            x1,y1,x2,y2,lprocr = cureplate
            imgh, imgw, _ = frame.shape
            platecx = ((x1+x2)/2) / imgw
            platecy = ((y1+y2)/2) / imgh
            pw = (x2-x1) / imgw
            ph = (y2-y1) / imgh
            outtxt = '0 ' + str(round(platecx,5)) + ' ' + str(round(platecy,5)) + ' ' + str(round(pw,5)) + ' ' + str(round(ph,5))
            print('GT: ', outtxt)
            gttxtfile.write(outtxt)  
        gttxtfile.close()
        #outtxt = outfilename + '\t' + outjsonstr + '\n'
        print(outgtfile)

    if DeleteLowconfPlates == 1:
         if minOcrConf < 0.9:
             dpos = imagepath.rfind('.')
             if dpos < 0:
                 continue
             labeltxtpath = imagepath[0:dpos] + '.txt'
             print('Deletion image: ', imagepath)
             print('Deletion label: ', labeltxtpath)
             DeletionCount = DeletionCount+1
             print('DeletionCount: ', DeletionCount, 'framecounter: ', framecounter)
             #os.remove(imagepath)
             #os.remove(labeltxtpath)
             #cv2.waitKey(0)
        
#results = model.predict(source="./data/db2/valid/images/",  save=True, imgsz=640) # Display preds. Accepts all YOLO predict arguments , , save=True stream=True,, device="cpu"
