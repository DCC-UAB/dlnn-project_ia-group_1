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
        else:
            path = os.path.join(dir_images_labels,'test.txt')
        
        with open(path, 'r') as file:
            self.samples = [tuple(line.split()) for line in file]

    def __len__(self):
        return (len(self.samples))

    def __getitem__(self, idx):


        img_name = os.path.join(self.dir_images, self.samples[idx][0]+'.jpg')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        target = torch.tensor(int(self.samples[idx][1]))

        return image, target


dir_images      = 'C:/Users/34644/Desktop/BUISNESS DETECTION/data/JPEGImages/'
dir_imgs_labels = 'C:/Users/34644/Desktop/BUISNESS DETECTION/data/'

input_size = 256

data_transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(input_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
data_transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_Set = ConTextDataset(dir_imgs_labels, dir_images, train = True,  transform = data_transforms_train)
test_Set  = ConTextDataset(dir_imgs_labels, dir_images, train = False, transform = data_transforms_test)

def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer