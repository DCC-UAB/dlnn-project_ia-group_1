# dlnn-project_ia-group_1
dlnn-project_ia-group_1 created by GitHub Classroom

In this GitHub repository, you can find all the necessary code of our implementation of a Business Classification problem, where given an image, our model is able to classify it to one of the different propsed classes.

In the dataset we can find 28 different business categories, which can be for instance: Bakery, Book Store, Motel and many others. As well as 24,255 images in total. In order to be able to obtain the images used for this task, you can click http://isis-data.science.uva.nl/jvgemert/images.tar.gz to directly download them.

In the preprocessing folder the code to separate train and test sets and give the label the images correspond to is provdied. Additionaly we have added the OCR_easyocr.py file, where we compute the Optic Character Recognition (OCR) for each image, * which improves the model significantly. (maybe explicar un pelin q es el q fa el OCR i pq millora) --> * which computes for each image ..., improving our model significantly.

When the data is preprocessed, in the OUR_utils folder, you can find the file utils.py, where the train and test loader are made, as well as the selection of the criterion and optimizer used. All the different other parameters like learning rate or the number of classes are set in the main.py file.

Then in the train.py, you can find the training of the model, with its scheduler and ...

In the test.py, the accuracy of the model in the test set is obtained, and a file with the weights of our model is saved, to later be able to visualize some results.

In the main.py, the configuration is set as well as the train and test are being performed.

Finally, in the Visualizations.ipyn, we load the weights obtained for our model, and compute some visualizations of the test images, with the true and predicted classes.





