import os
import keras_ocr

def image_information(train=True):
    if train:
        variable = 'train'
    else:
        variable = 'test'

    reader = keras_ocr.pipeline.Pipeline()
    with open(f'/home/xnmaster/data/{variable}.txt', 'r') as file:
        images_name = file.readlines()

    images = [os.path.join('/home/xnmaster/data/JPEGImages/', image_name.split()[0] + '.jpg') for image_name in images_name]

    results = []
    for image_path in images:
        output = reader.recognize([image_path])
        result = " ".join([ind[0] for ind in output[0]]) if len(output) != 0 else '0'
        results.append(result)

    return results


def obtain_keras_ocr(train=True):
    if train:
        variable = 'train'
    else:
        variable = 'test'
    results = image_information(train)
    with open(f'/home/xnmaster/data/keras_ocr_{variable}.txt', 'w') as file:
        file.write('\n'.join(results))

obtain_keras_ocr(train=True)
obtain_keras_ocr(train=False)