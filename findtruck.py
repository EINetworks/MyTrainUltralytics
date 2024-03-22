import csv
import cv2

folderpath = '/mnt/ssd2/DATASET/LPR/AnkitTollLPRData/190324/'
CsvPath= '/mnt/ssd2/DATASET/LPR/AnkitTollLPRData//19March24.csv'
# Open the CSV file
with open(CsvPath) as file_obj:
    # Create a reader object
    reader_obj = csv.reader(file_obj)
    counter = 0
    for row in reader_obj:
        if(len(row[1]) > 0):
            if row[20] == 'Bus' or row[20] == 'Truck' or row[17] == 'Bus' or row[17] == 'Truck' or 'Axle' in row[17] or 'Axle' in row[20]:
                print(counter, ': ', row[1], ', 17: ', row[17], ', 20: ', row[20])
                imgname = folderpath + row[1] + '.Jpg'
                frame = cv2.imread(imgname)
                if frame is None:
                    continue
                print(imgname)
                cv2.imshow("frame", frame)
                cv2.waitKey(10)
                counter = counter + 1

