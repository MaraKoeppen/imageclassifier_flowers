# Importing necessary libraries for data handling, model building, and visualization

import matplotlib.pyplot as plt

import torch #Core PyTorch libraries for model building and optimization.
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image  #  For image processing
import argparse # For parsing command-line arguments

import json  # For loading JSON files Load the prepared mapping from category labels to flower names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Setting up argument parsing for configurable parameters

parser = argparse.ArgumentParser(description="Training script for image classification model")
parser.add_argument('--data_dir', type=str, default='flowers', help="Path to the data directory default is 'flowers'")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders, def 64')
parser.add_argument('--model', type=str, choices=['vgg16', 'densenet121'], default='vgg16', help='Choose model architecture: vgg16 or densenet121')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer, default is 0.001')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (default: 3)')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier layer (default: 512)')
parser.add_argument('--checkpoint_dir', type=str, default='./', help='Directory to save the model checkpoint, default is the current directory')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to use: cpu or cuda (default: cuda)')
args = parser.parse_args()


# GPU availability
device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
print(f"Using {device} for training.")


# Load data folder specified by the user
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), #Randomly crops and resizes images to 224x224 for data augmentation.
                                       transforms.RandomHorizontalFlip(), #Randomly flips images horizontally to reduce overfitting.
                                       transforms.ToTensor(), #converts image to pytorchTensor + It also scales the pixel values from the typical 0–255 range (for 8-bit images) to the 0–1 
                                       transforms.Normalize([0.485, 0.456, 0.406],  
                                                            [0.229, 0.224, 0.225])]) # Adjusts the pixel values based on the statistical properties (mean and standard deviation) of the dataset the model was trained on (usually ImageNet).

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder and applying the transformations from above
train_data = datasets.ImageFolder(args.data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(args.data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(args.data_dir + '/valid', transform=valid_transforms)

# Creating DataLoaders for batching and shuffling the data batch size and shuffling has an impact on the model quality and performance
trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size)


# Freeze convolutional layers
def setup_vgg16(hidden_units):
    model = models.vgg16(pretrained=True) # pytorch loads pretrained model vgg16 from www, including pretrained weights trained on imagenet
    for param in model.features.parameters():
        param.requires_grad = False # Freezes convolutional(features) layers to prevent updating pre-trained weights. (colors, textures and edges)
    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),  # takes over the number of features of the last c layer to the fully connected/classification layer and specifies the output the hidden_units-value to adjust for performance)
        nn.ReLU(), # activation function to process non linear realationships more complex ones
        nn.Dropout(0.5), # is a technique to prevent overfitting. It works by randomly "deactivating" some neurons during training (setting them to 0), forcing the model to learn more robust features.
        nn.Linear(hidden_units, 102),  # Verwendet den gleichen hidden_units-Wert
        nn.LogSoftmax(dim=1) # computes propability
    )
    model.classifier = classifier
    return model

def setup_densenet121(hidden_units):
    model = models.densenet121(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(
        nn.Linear(1024, hidden_units),  # Verwendet den übergebenen hidden_units-Wert
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),  # Verwendet den gleichen hidden_units-Wert
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    return model

      

# Select model based on user input
if args.model == 'vgg16':
    model = setup_vgg16(args.hidden_units)
elif args.model == 'densenet121':
    model = setup_densenet121(args.hidden_units)

print(f"Using {args.model} model for training.")

#training
criterion = nn.NLLLoss() # Negative Log-Likelihood Loss, commonly used for classification with log probabilities
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate) # Adam optimizer adjusts the learning rate automatically for each parameter(weights and biases)
model.to(device) # Moving the model to the chosen device (CPU/GPU)

# Initializing variables for training
epochs = args.epochs  # Number of epochs specified by the user
steps = 0 # Tracks the number of batches processed
running_loss = 0 # Accumulates training loss to monitor performance
print_every = 5 # Defines how often to print training progress , currently not used in the training, results are only printed after each epoch to improve the performance

# Initializing variables for early stopping, sometimes the model is not inproving over the epochs this function would stop the training if this was the case
best_loss = float('inf')
epochs_no_improve = 0
early_stop_patience = 3  # stops training if no improvement in test loss for 3 consecutive epochs are used to avaid overfitting (default epochs is also 3 in args)

# Training loop forward pass going through all pictures/data and than starts the back propagation to correct the weights and biases to improve the prediction
for epoch in range(epochs):  # Loop over the number of epochs /the model is going through the complete training data for each epoch
    for inputs, labels in trainloader:
        steps += 1 # Increment step counter
        
        
        inputs, labels = inputs.to(device), labels.to(device) # Move inputs and labels to the chosen device
        
        # Forward pass through the model
        logps = model.forward(inputs)
        loss = criterion(logps, labels) # Compute the loss
        
        # Backpropagation
        optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients
        optimizer.step() # Update weights
        
        running_loss += loss.item() # Add the current loss to the running total
        
  # Validation phase after each epoch  
    test_loss = 0
    accuracy = 0
    model.eval() # Set model to evaluation mode (disables dropout, batchnorm, etc.)
    with torch.no_grad(): # Disable gradient computation for validation (saves memory and computations)
        for inputs, labels in testloader: # Loop over the validation data
            inputs, labels = inputs.to(device), labels.to(device) 
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels) # Compute validation loss
            
            test_loss += batch_loss.item() # Accumulate validation loss
            
            # Calculating accuracy after each epoch
            ps = torch.exp(logps) # Convert log probabilities to probabilities
            top_p, top_class = ps.topk(1, dim=1) # Get the class with the highest probability
            equals = top_class == labels.view(*top_class.shape) # Check if predictions are correct
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Calculate mean accuracy

     # Calculate average validation loss and accuracy
    avg_test_loss = test_loss / len(testloader)
    avg_accuracy = accuracy / len(testloader)

    # Print training and validation statistics
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/steps:.3f}.. "
          f"Test loss: {avg_test_loss:.3f}.. "
          f"Test accuracy: {avg_accuracy:.3f}")
    
    running_loss = 0  # Reset running loss for the next epoch
    model.train()  # Switch back to training mode

    # Early Stopping Stops training if there's no improvement in test loss over consecutive epochs.
    if avg_test_loss < best_loss:
        best_loss = avg_test_loss # Update best loss
        epochs_no_improve = 0  # Reset counter if there is improvement
    else:
        epochs_no_improve += 1 # Increment counter if no improvement

    
    if epochs_no_improve >= early_stop_patience: # Check if early stopping condition is met
        print("Early stopping triggered!")
        break

# create classification dictionary folder number / name of flower
model.class_to_idx = train_data.class_to_idx

# # Saving the trained model's checkpoint for use in prediction in pth file
checkpoint = {
    'epoch': epochs,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'class_to_idx': model.class_to_idx,
    'classifier': model.classifier,
}

# Dynamic checkpoint filename based on the model architecture
checkpoint_filename = f"{args.checkpoint_dir}model_checkpoint_{args.model}.pth"

# save dynamic checkpoint filename based on the model architecture
torch.save(checkpoint, checkpoint_filename)

print(f"Checkpoint saved as {checkpoint_filename}!")