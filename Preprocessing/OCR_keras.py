#These file usses the keras ocr, it follows the same principle as the OCR_eassy.py file. It just change how we create the reader object. 
import os
import keras_ocr
from tqdm import tqdm

def obtain_keras_ocr(train=True):
    if train:
        variable = 'train'
    else:
        variable = 'test'

    reader = keras_ocr.pipeline.Pipeline()
    with open(f'/home/xnmaster/data/{variable}.txt', 'r') as file:
        images_name = file.readlines()

    images = [os.path.join('/home/xnmaster/data/JPEGImages/', image_name.split()[0] + '.jpg') for image_name in images_name]
    size_images = len(images)

    with open(f'/home/xnmaster/data/keras_ocr_{variable}.txt', 'w') as ocr_file:
        for idx, image_path in enumerate(tqdm(images, desc=f'Processing {variable} images', unit='image')):
            output = reader.recognize([image_path])

            ocr_output = " ".join([ind[0] for ind in output[0]]) if len(output) != 0 else '0'
            ocr_file.write(ocr_output + '\n' if idx < size_images else ocr_output)   #Write the contents at each line. The last image don't add the \n

    print(f"Done {variable} ocr text")


obtain_keras_ocr(train=True)
obtain_keras_ocr(train=False)
