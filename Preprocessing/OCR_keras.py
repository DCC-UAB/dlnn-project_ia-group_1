import cv2
import os
from tqdm import tqdm
import keras_ocr

def obtain_keras_ocr(reader, train=True):
    if train:
        variable = 'train'
    else:
        variable = 'test'

    images_directory = '/home/xnmaster/data/JPEGImages/'

    with open(f'/home/xnmaster/data/{variable}.txt', 'r') as file:
        images_name = file.readlines()

    images = [os.path.join(images_directory, image_name.split()[0] + '.jpg') for image_name in images_name]
    size_images = len(images)

    with open(f'/home/xnmaster/data/keras_ocr_{variable}.txt', 'w') as ocr_file:
        for idx, image_path in enumerate(tqdm(images, desc=f'Processing {variable} images', unit='image')):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 480))  # Resize the image to (640, 480) or another suitable size
            
            output = reader.recognize([image])

            ocr_output = " ".join([ind[0] for ind in output[0]]) if len(output) != 0 else '0'
            ocr_file.write(ocr_output + '\n' if idx < size_images else ocr_output)   #Write the contents at each line. The last image doesn't add the \n

    print(f"Done {variable} ocr text")

reader = keras_ocr.pipeline.Pipeline()

obtain_keras_ocr(reader, train=True)
obtain_keras_ocr(reader, train=False)
