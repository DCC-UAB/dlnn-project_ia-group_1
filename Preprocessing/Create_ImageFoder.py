import os
import shutil
from torchvision import datasets, models, transforms

business_dict = {
    1: "Bakery",
    2: "Barber",
    3: "Bistro",
    4: "Bookstore",
    5: "Cafe",
    6: "ComputerStore",
    7: "CountryStore",
    8: "Diner",
    9: "DiscountHouse",
    10: "Dry Cleaner",
    11: "Funeral",
    12: "Hotspot",
    13: "MassageCenter",
    14: "MedicalCenter",
    15: "PackingStore",
    16: "PawnShop",
    17: "PetShop",
    18: "Pharmacy",
    19: "Pizzeria",
    20: "RepairShop",
    21: "Restaurant",
    22: "School",
    23: "SteakHouse",
    24: "Tavern",
    25: "TeaHouse",
    26: "Theatre",
    27: "Tobacco",
    28: "Motel"
}

def directori_creation(business_dict, train = True):
    # Define the directory and subdirectory names
    if train:
        directory_name = '/home/xnmaster/data/Train_Images/'
    else:
        directory_name = '/home/xnmaster/data/Test_Images/'

    for buisness in business_dict.values():
        subdirectory_name = buisness

        # Define the full path to the subdirectory
        subdirectory_path = os.path.join(directory_name, subdirectory_name)

        # Create the subdirectory
        # The argument exist_ok=True prevents an error if the directory already exists
        os.makedirs(subdirectory_path, exist_ok=True)


def move_images(business_dict, train = True):
    # Define the image path and the destination subdirectory path
    if train:
        path = '/home/xnmaster/data/train.txt'
        directory_name = '/home/xnmaster/data/Train_Images/'
    else:
        path = '/home/xnmaster/data/test.txt'
        directory_name = '/home/xnmaster/data/Test_Images/'

    with open(path, 'r') as file:
        image_name_label = file.readlines()
    
    for image_label in image_name_label:
        image_name, label = image_label.rstrip().split()

        image_path            = f'/home/xnmaster/data/JPEGImages/{image_name}' + '.jpg'
        destination_directory = os.path.join(directory_name, business_dict[label])

        # Move the image to the destination directory
        shutil.move(image_path, destination_directory)

#Creation of two folder containing the train and test images. Each folder contains the same subfolders, one for each class in the dataset.
directori_creation(business_dict, train = True)
directori_creation(business_dict, train = False)


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/home/xnmaster/data/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['Train_Images', 'Test_Images']}