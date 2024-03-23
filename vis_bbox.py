import os
import cv2
import argparse

#python vis_bbox.py --folder_path data/train/images/ --classes temp.classes --save_path testout/

def main():

    parser = argparse.ArgumentParser(description='Verify Yolo Annotations')
    parser.add_argument('--folder_path', type=str, help='Path to folder containing images and the corresponding labels')
    parser.add_argument('--classes', type=str, help='Path to classes.names file')
    parser.add_argument('--save_path', type=str, help='Path to save annotated images')
    args = parser.parse_args()

    with open(args.classes) as class_names:
        classes = class_names.readlines()
    
    classes = [x.strip() for x in classes]
    
    class_names = {}
    for idx,name in enumerate(classes):
        class_names[name] = idx
    
    class_names = {v: k for k, v in class_names.items()}
    
    print(class_names)

    args = parser.parse_args()

    os.makedirs(args.save_path,exist_ok=True)

    all_files = os.listdir(args.folder_path)
    annotations, images = [], []

    for file in all_files:
        if '.txt' in file:
            annotations.append(file)
        elif '.jpg' or '.jpeg' or '.png' in file:
            images.append(file)
    print(images)
    if len(images) > len(annotations):
        print("Some images dont have annotations", len(images), len(annotations))
        return

    elif len(images) < len(annotations):
        print("There are more annotations than images")
        return

    images.sort()
    annotations.sort()

    color = (0,0,255)
    counter = 0
    skipcounter = 1
    StartCounter = 5000
    TotalCount = 0
    WrongPlateCount = 0
    for (image, annotation) in zip(images, annotations):
        print(counter, ': ', image)
        TotalCount = TotalCount + 1
        counter = counter + 1
        if counter%skipcounter != 0:
            continue
        if counter < StartCounter:
            continue
        img = cv2.imread(os.path.join(args.folder_path,image))
        height, width, _ = img.shape
        with open(os.path.join(args.folder_path,annotation)) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        #newAnnotation = []
        wrongPlate = 0
        for annot in content:
            annot = annot.split()
            class_idx = int(annot[0])
            x,y,w,h = float(annot[1]),float(annot[2]),float(annot[3]),float(annot[4])
            #
            xmin = int((x*width) - (w * width)/2.0)
            ymin = int((y*height) - (h * height)/2.0)
            xmax = int((x*width) + (w * width)/2.0)
            ymax = int((y*height) + (h * height)/2.0)
            print(x,y,w,h, width, height, xmin, ymin, xmax, ymax)
            #newAnnotation.append((class_idx, x,y,2*w,2*h))

            color = (0,0,255)
            if w > 0.7 or h > 0.7 or (xmax-xmin) < 50 or (ymax-ymin) < 20:
                wrongPlate = 1
                color = (255,0,0)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img, class_names[int(class_idx)], (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 3)

        if 0: #wrongPlate==1:
            WrongPlateCount = WrongPlateCount+1
            imagefilepath = os.path.join(args.folder_path,image)
            txtfilepath = os.path.join(args.folder_path,annotation)
            print('Deleting: ', imagefilepath)
            os.remove(imagefilepath)
            os.remove(txtfilepath)
        print('WrongPlateCount: ', WrongPlateCount, 'TotalCount: ', TotalCount)
        cv2.imshow("GT", img)
        cv2.waitKey(0)
        if 0:  ## to fix any mistakes in GT file
            gttxtfile = open(os.path.join(args.folder_path,annotation),'w') 
            for annot in newAnnotation:
                classid, platecx, platecy, pw, ph = annot
                outtxt = '0 ' + str(round(platecx,5)) + ' ' + str(round(platecy,5)) + ' ' + str(round(pw,5)) + ' ' + str(round(ph,5))
                print('GT: ', outtxt)
                gttxtfile.write(outtxt)  
            gttxtfile.close()

        #cv2.imwrite(os.path.join(args.save_path,image), img)
        print("saving image",image)
    print('WrongPlateCount: ', WrongPlateCount, 'TotalCount: ', TotalCount)
    print("Done")


if __name__ == '__main__':
	main()
			
