import os
import easyocr

def obtain_ocr(train = True):
    if train:
        variable = 'train'
    else:
        variable = 'test'

    ocr_file_path  = f'/home/xnmaster/data/ocr_{variable}.txt'
    path_1         = f'/home/xnmaster/data/{variable}.txt'
    dir_images     = '/home/xnmaster/data/JPEGImages/'
    with open(ocr_file_path, 'w') as ocr_file, open(path_1, 'r') as train_file:
        for line in train_file:
            img_name = line.split()[0]
            file_path = os.path.join(dir_images, img_name + '.jpg')
            ocr_output = reader.readtext(file_path, detail = 0, paragraph= True)
            if len(ocr_output) == 0:
                ocr_output = '0'
            else:
                ocr_output = ocr_output[0]
            ocr_file.write(ocr_output + '\n')

reader = easyocr.Reader(['en'], gpu=True)
obtain_ocr(reader, train = True)
obtain_ocr(reader, train = False)