#These file obtains the OCR for each of the images in the train and test set

import os
import easyocr

def obtain_ocr(reader, train = True):
    if train:
        variable = 'train'
    else:
        variable = 'test'

    ocr_file_path  = f'/home/xnmaster/data/Eassyocr_{variable}.txt'  #The ocr output will be saved at ocr_{variable}.txt
    with open(f'/home/xnmaster/data/{variable}.txt', 'r') as file:
        images_name = file.readlines()
    images = [os.path.join('/home/xnmaster/data/JPEGImages/', image_name.split()[0] + '.jpg') for image_name in images_name] #We read from {variable}.txt the names of each of the train and test sets
    size_images = len(images)
    with open(ocr_file_path, 'w') as ocr_file: # Create the ocr ouput file
        for idx, image_path in enumerate(images):                                    #For each of the images
            ocr_output = reader.readtext(image_path, detail = 0, paragraph= True)    #Obtain the ocr output
            ocr_output = ocr_output[0] if len(ocr_output) != 0 else '0'              #Write the output if it has find text else write 0
            ocr_file.write(ocr_output + '\n' if idx < size_images else ocr_output)   #Write the contents at each line. The last image don't add the \n

reader = easyocr.Reader(['en'], gpu=True)
obtain_ocr(reader, train = True)
obtain_ocr(reader, train = False)