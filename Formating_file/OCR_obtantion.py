import os
import easyocr

def obtain_ocr(reader, train = True):
    if train:
        variable = 'train'
    else:
        variable = 'test'

    ocr_file_path  = f'/home/xnmaster/data/ocr_{variable}.txt'
    new_path_train = f'/home/xnmaster/data/{variable}.txt'
    dir_images      = '/home/xnmaster/data/JPEGImages/'
    with open(ocr_file_path, 'w') as ocr_file, open(new_path_train, 'r') as train_file:
        for line in train_file:
            img_name, label = line.split()
            file_path = os.path.join(dir_images, img_name + '.jpg')
            ocr_output = ''
            for text in reader.readtext(file_path):
                ocr_output += ' ' + text[1]
            if len(ocr_output) < 2:
                ocr_file.write('0' + '\n')
            else:
                ocr_file.write(ocr_output + '\n')

#reader = easyocr.Reader(['en'], gpu=True)
#obtain_ocr(reader, True)
#obtain_ocr(reader, False)