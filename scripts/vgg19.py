import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import cv2
import albumentations as A
import torchvision.models as models
import os
from tqdm import tqdm
import time
import skimage.io as skio
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")


artworks = pd.read_csv('artworks.csv')

artworks['s3_path'] = artworks.apply(lambda row: row['style'] \
                                     + "/" + row['image'].split('/')[-1].split('.')[0] + ".jpg", 
                                     axis=1)

# Define transformations
transforms = A.Compose([
    A.Flip(p=0.5),
    A.Rotate(limit=10, 
             border_mode=cv2.BORDER_CONSTANT, 
             value=0.0, p=0.75),
    A.RandomResizedCrop(width=224, height=224, scale=(0.5, 1), p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225), 
                max_pixel_value=255.0, 
                p=1.0)
])


class ArtDataset(Dataset):
    def __init__(self, df, label_dict, transforms, fs= None):
        self.df = df
        self.transforms = transforms
        self.label_dict = label_dict
        self.fs = fs
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Get filename and label
        filename = row['s3_path']
        #label = torch.zeros(25, dtype = torch.long)
        label = torch.tensor(label_dict[row['style']], dtype = torch.long)
        # Read image, correct color channels
        img = self.load_img(filename)
#        print(img)
        # adding this portion if the image has 4 channels or more -- Chandrish
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis = 2)
            img = np.repeat(img, 3, axis = 2)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        # Augmentations + normalization
        transformed = self.transforms(image=img.astype(np.uint8))
        img = transformed['image']
        
        img = img.transpose(2, 0, 1)
        # Convert to tensor
        img = torch.tensor(img).float()
        #img = torch.permute(2, 0, 1)
        return img, label
    
    def load_img(self, s3_path):
        try:
            img_arr = skio.imread(s3_path)
            img_arr.shape
        except:
            img_arr = skio.imread('symbolism/baroness-fernand-van-der-bruggen-1900.jpg')
        return img_arr

    
def set_classification_layer(model, model_type='vgg', num_classes=25):
    if model_type == 'vgg':
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )
    elif model_type == 'resnet':
        model.fc = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    elif model_type == 'vit':
        model.heads = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    elif model_type == 'convnext':
        model.classifier = nn.Sequential(
            nn.LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=768, out_features=num_classes, bias=True)
        )
    else:
        print(f'Unknown model_type {model_type}. Acceptable types are: "vgg", "resnet", "vit", or "convnext"')   

def freeze_model(model, **classargs):
    '''
    Given an existing model, freeze pre-trained weights and
    re-instantiate the classifier.
    '''
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Re-instantiate the classifier head
    model = set_classification_layer(model, **classargs)



def eval_model(model, dl, training_params):
    # Get GPU if available
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # Evaluate
    model.eval()
    # model = model.to(device)
    total_loss = 0
    total_obs = 0
    total_correct = 0
    loss_fct = training_params['loss_fct']
    for X, y in dl:
        n_obs = len(y)
        # Forward pass and calculate loss
        yhat = model(X.to(device))#.softmax(dim=1)
        loss = loss_fct(yhat.to(device), y.to(device))
        total_loss += n_obs * loss.item()
        total_obs += n_obs
        # Calculate batch accuracy
        ypred = np.argmax(yhat.cpu().detach().numpy(), axis=1)
        y_arr = y.detach().numpy()
        total_correct += n_obs * accuracy_score(y_arr, ypred)
    # Return loss, accuracy
    avg_loss = total_loss / total_obs
    accuracy = total_correct / total_obs
    return avg_loss, accuracy
    
    
def train_model(model, optimizer, scheduler, train_ds, valid_ds, training_params):
    # Get loss function
    loss_fct = training_params['loss_fct']
    # Create dataloaders based on batch size
    batch_size = training_params['batch_size']
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    # Get GPU if available
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # Train
    for _ in range(training_params['epochs']):
        # Put model in train mode
        model.train()
        # Train on training dataloader
        for X, y in tqdm(train_dl):
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass and loss calculation
            yhat = model(X.to(device))#.softmax(dim=1)
            loss = loss_fct(yhat.to(device), y.to(device))
            # Backward pass and step
            loss.backward()
            optimizer.step()
        scheduler.step()  # update scheduler each epoch
        # Calculate loss, accuracy on train and validation
        train_loss, train_acc = eval_model(model, train_dl, training_params)
        valid_loss, valid_acc = eval_model(model, valid_dl, training_params)
        train_str = f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f}"
        valid_str = f" | valid loss: {valid_loss:.4f} | valid acc: {valid_acc:.4f}"
        print(f'[{_}] ' + train_str + valid_str)
        torch.save(model.state_dict(), f'models/vgg19_epoch{_}.pth')

        
# Dictionary for easily passing training arguments
training_params = {'epochs': 20,
                  'batch_size': 16,
                  'loss_fct': nn.CrossEntropyLoss()}


df = artworks.sample(frac = 1, random_state = 62).reset_index(drop = True)
split1 = int(0.7 * df.shape[0])
split2 = int(0.85 * df.shape[0])
train_df, valid_df, test_df = df.iloc[:split1].copy(), df.iloc[split1: split2].reset_index(drop = True), \
                                    df.iloc[split2:].reset_index(drop = True)

label_dict = {style: i for i, style in enumerate(sorted(artworks['style'].unique()))}
# creating Datasets
train_ds = ArtDataset(train_df, label_dict, transforms)
valid_ds = ArtDataset(train_df, label_dict, transforms)
test_ds = ArtDataset(train_df, label_dict, transforms)

# Trying VGG 19
from torchvision.models import vgg19
model = vgg19(pretrained = True)

# freezing the parameters
freeze_model(model, num_classes=25, model_type='vgg')

# training
from torch.optim.lr_scheduler import StepLR
optimizer = optim.Adam(model.parameters(), )
scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
training_params = {'epochs': 10,
                  'batch_size': 256,
                  'loss_fct': nn.CrossEntropyLoss()}

train_model(model, optimizer, scheduler, train_ds, valid_ds, training_params)


