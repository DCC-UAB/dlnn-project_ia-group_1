#These file obtains the OCR for each of the images in the train and test set
import os
import easyocr

def obtain_ocr(directory_test_train_files, images_directory, reader, train = True):
    if train:
        variable = 'train.txt'
        ocr_file = 'Eassyocr_train.txt'

    else:
        variable = 'test.txt'
        ocr_file = 'Eassyocr_test.txt'

    ocr_file_path = os.path.join(directory_test_train_files, ocr_file)                  #Path where we will stored the output of the eassy ocr tool
    directory_test_train_files = os.path.join(directory_test_train_files, variable)     #Path where we will read the name of the images

    with open(directory_test_train_files, 'r') as file:
        images_name = file.readlines()

    images = [os.path.join(images_directory, image_name.split()[0] + '.jpg') for image_name in images_name] #We read from {variable}.txt the names of each of the train and test sets
    size_images = len(images)
    with open(ocr_file_path, 'w') as ocr_file: # Create the ocr ouput file
        for idx, image_path in enumerate(images):                                    #For each of the images
            ocr_output = reader.readtext(image_path, detail = 0, paragraph= True)    #Obtain the ocr output
            ocr_output = ocr_output[0] if len(ocr_output) != 0 else '0'              #Write the output if it has find text else write 0
            ocr_file.write(ocr_output + '\n' if idx < size_images else ocr_output)   #Write the contents at each line. The last image don't add the \n

#If you want to try the code you need to put the directory where you have stored the test.txt and train.txt files obtained form the Train_and_Test.py file.
#Also you need to specify the directory containing the .jpg images
directory_test_train_files = '/home/xnmaster/data/'                         #Directory containing the name of the images for the train and test set as well as the classes of each of the images
images_directory           = '/home/xnmaster/data/JPEGImages/'              #Directory containing the images .jpg format

reader = easyocr.Reader(['en'], gpu=True)
obtain_ocr(directory_test_train_files, images_directory, reader, train = True)
obtain_ocr(directory_test_train_files, images_directory, reader, train = False)