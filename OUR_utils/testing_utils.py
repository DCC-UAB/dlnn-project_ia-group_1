from OUR_models.our_models import *
import wandb
import torch
import torch.nn 

config = dict(
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=5e-3,
    dataset="MNIST")

class ConTextDataset(Dataset):
    def __init__(self,dir_images_labels, dir_images, train = True, transform = None):
        self.dir_images        = dir_images
        self.dir_images_labels = dir_images_labels
        self.transform = transform

        if train:
            path = os.path.join(dir_images_labels,'train.txt')
            path_ocr = os.path.join(dir_images_labels, 'ocr_train.txt')
        else:
            path = os.path.join(dir_images_labels,'test.txt')
            path_ocr = os.path.join(dir_images_labels, 'ocr_test.txt')
        
        with open(path, 'r') as file, open(path_ocr, 'r') as ocr_File:
            self.samples = [tuple(line.split(), line_ocr) for line, line_ocr in zip(file, ocr_File)]

    def __len__(self):
        return (len(self.samples))

    def __getitem__(self, idx):


        img_name = os.path.join(self.dir_images, self.samples[idx][0]+'.jpg')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        if self.samples[idx][2] != '0':
            #Implement text embedding
            pass
        target = torch.tensor(int(self.samples[idx][1]))

        return image, target

def get_transform(train = True):
    input_size = 256
    if train:
        data_transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(input_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        return data_transforms_train
    else:
        data_transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        return data_transforms_test


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

def make(config, device="cuda"):
    # Make the data
    dir_imgs_labels = '/home/xnmaster/data/'
    dir_images      = '/home/xnmaster/data/JPEGImages/'

    train = ConTextDataset(dir_imgs_labels, dir_images, train = True,  transform = get_transform(train = True))
    test  = ConTextDataset(dir_imgs_labels, dir_images, train = False, transform = get_transform(train = False))

    train_loader  = make_loader(train, batch_size=config.batch_size)
    test_loader   = make_loader(test, batch_size=config.batch_size)

    model = ConTextTransformer()
    # Make the model
    #model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer