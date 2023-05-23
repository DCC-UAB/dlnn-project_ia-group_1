import os
import keras_ocr

def image_information(train=True):
    if train:
        variable = 'train'
    else:
        variable = 'test'

    reader = keras_ocr.pipeline.Pipeline()
    batch_size = 100
    start_index = 0
    with open(f'/home/xnmaster/data/{variable}.txt', 'r') as file:
        images_name = file.readlines()

    images = [os.path.join('/home/xnmaster/data/JPEGImages/', image_name.split()[0]) for image_name in images_name]
    size_images = len(images)

    while start_index < size_images:
        images_to_read = images[start_index:min(start_index + batch_size, size_images)]
        output = reader.recognize(images_to_read)
        start_index += batch_size
        yield output


def obtain_keras_ocr(train = True):
    if train:
        variable = 'train'
    else:
        variable = 'test'
    with open(f'/home/xnmaster/data/keras_ocr_{variable}.txt', 'w') as file:
        for output in image_information(train):
            result_to_write  = [" ".join([ind[0] for ind in image_output]) if len(image_output) != 0 else '0' for image_output in output]
            file.write('\n'.join(result_to_write))