import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import pandas as pd
import csv
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

# Ensure CBAM is correctly imported
from cbam import CBAM  # Assuming CBAM is implemented elsewhere

# Define the CXRDataset class
class CXRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df (DataFrame): The DataFrame containing the data split.
            img_dir (str): Directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform
        self.path_to_images = img_dir

        self.PRED_LABEL = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = str(self.df.iloc[idx, 0])  # Assuming first column is "Image Index"
        image_path = os.path.join(self.path_to_images, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found in directory.")
            return None  # Skip if image is missing

        image = Image.open(image_path)
        image = image.convert('RGB')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i, condition in enumerate(self.PRED_LABEL):
            if self.df[condition].iloc[idx] > 0:
                label[i] = 1

        label = label.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label, image_name


# Define the DenseNet with CBAM model
class DenseNetWithCBAM(nn.Module):
    def __init__(self, num_classes=14, reduction=16, kernel_size=7):
        super(DenseNetWithCBAM, self).__init__()

        # Load pre-trained DenseNet-121
        self.densenet = models.densenet121(pretrained=True)

        # Replace the classifier part
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

        # Adding CBAM after each convolution block
        self.cbam = CBAM(in_channels=1024, reduction=reduction, kernel_size=kernel_size)

    def forward(self, x):
        # Pass through DenseNet convolution layers
        x = self.densenet.features(x)

        # Apply CBAM attention
        x = self.cbam(x)

        # Global average pooling (this is important to reduce dimensions before fully connected layer)
        x = torch.mean(x, dim=(2, 3))  # Global average pooling across height and width

        # Pass through the classifier
        x = self.densenet.classifier(x)
        return x

def checkpoint(model, best_loss, epoch, LR):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, 'results/checkpoint')
# Training function with checkpointing and learning rate decay
def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: DenseNetWithCBAM model
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained model
        best_epoch: epoch on which best model val loss was obtained
    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch

def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
    
    # Flatten the lists
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate AUC for each label (class) separately
    auc = []
    for i in range(all_labels.shape[1]):
        try:
            auc_score = roc_auc_score(all_labels[:, i], all_predictions[:, i])
            auc.append(auc_score)
        except ValueError:
            # Handle case where only one class is present in the true labels
            auc.append(np.nan)  # Use NaN if AUC cannot be computed
    
    auc = np.nanmean(auc)  # Take the average AUC over all classes

    # Calculate F1 score (using a threshold of 0.5 for binary classification)
    f1 = f1_score(all_labels, (all_predictions > 0.3).astype(int), average='macro', zero_division=0)

    return auc, f1

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def split_dataset(merged_csv, test_size=0.1, val_size=0.1, random_state=42):
    # Load the merged dataset CSV
    df = pd.read_csv(merged_csv, encoding='ISO-8859-1')
    
    # Split into train and remaining (val + test)
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=random_state, shuffle=True)
    
    # Split remaining into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=val_size / (test_size + val_size), random_state=random_state, shuffle=True)
    
    return train_df, val_df, test_df
# Initialize model
model = DenseNetWithCBAM(num_classes=14)  # Adjust the number of classes as needed
model = model.cuda()  # Move the model to GPU if available

# Define optimizer and loss function
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Load the dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_df, val_df, test_df = split_dataset('merged_images_labels.csv')

# Save the splits to CSV files (optional)
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)
test_df.to_csv('test_split.csv', index=False)
# Step 1: Create train, validation, and test datasets
train_dataset = CXRDataset(df=train_df, img_dir='starter_images', transform=transform)
val_dataset = CXRDataset(df=val_df, img_dir='starter_images', transform=transform)
test_dataset = CXRDataset(df=test_df, img_dir='starter_images', transform=transform)

# Step 2: Create DataLoader objects for each dataset
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 3: Calculate dataset sizes
dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset), 'test': len(test_loader.dataset)}

# Step 2: Create the DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Step 3: Calculate dataset sizes
dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}

# Step 4: Define weight decay (L2 regularization)
weight_decay = 1e-4  # You can adjust this value based on your experimentation

# Step 5: Pass the dataloaders, dataset sizes, and weight decay to the training function
train_model(
    model,
    criterion,
    optimizer,
    LR=0.001,
    num_epochs=10,
    dataloaders={'train': train_loader, 'val': val_loader},
    dataset_sizes=dataset_sizes,
    weight_decay=weight_decay
)
auc_chexnet_cbam, f1_chexnet_cbam = evaluate_model(model, val_loader)
params_chexnet_cbam = count_parameters(model)

ablation_results = pd.DataFrame({
    'Model Architecture': ['CheXNet + CBAM'],
    'AUC': [auc_chexnet_cbam],
    'F1 Score': [f1_chexnet_cbam],
    'Parameters (Millions)': [params_chexnet_cbam / 1e6]
})

# Output the ablation table
print(ablation_results)