#Prediction based on trained model (vg116 and Densenet121)
#commandline used for testing
#%run predict.py ./model_checkpoint_densenet121.pth ./flowers/valid/100/image_07895.jpg location of image file can be changed
#or
#% python run predict.py ./model_checkpoint_vgg16.pth ./image_1.jpg
python predict.py ./model_checkpoint_vgg16.pth ./image_1.jpg



# imports
import torch
from torchvision import models
import torch.optim as optim
import json
from PIL import Image
from torchvision import transforms
import argparse


parser = argparse.ArgumentParser(description="Image Classifier with a trained model")
parser.add_argument('checkpoint', type=str, help="Path to the trained model (.pth file")
parser.add_argument('image', type=str, help="Path to the image file for prediction")
parser.add_argument('--gpu', action='store_true', help="Use GPU if available")#default is cpu gpu can be selected
args = parser.parse_args()

# give user feedback of model, image and gpu vs. cpu usage
print(f"Checkpoint path: {args.checkpoint}")
print(f"Image path: {args.image}")
print(f"Use GPU: {args.gpu}")
input("Arguments parsed. Press Enter to continue.")

# Load category to name mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

checkpoint_path = args.checkpoint
image_path = args.image

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath, map_location=device)
        
        if 'densenet121' in filepath.lower():
            model = models.densenet121(pretrained=True)
        elif 'vgg16' in filepath.lower():
            model = models.vgg16(pretrained=True)
        else:
            raise ValueError("Unknown model type in checkpoint filename")
        
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        
        model.eval()  # Set model to evaluation mode
        return model.to(device), optimizer, checkpoint['epoch']
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None

# Load model
model, optimizer, epochs = load_checkpoint(checkpoint_path)
if model:
    print(f"Checkpoint successfully loaded from: {checkpoint_path}")

def process_image(image_path):
    pil_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(pil_image)

def predict(image_path, model, topk=3):
    image = process_image(image_path).unsqueeze(0).to(device)  # Add batch dimension
    model = model.to(device)
    
    with torch.no_grad():
        output = model(image)
    
    probabilities = torch.nn.functional.softmax(output, dim=1)
    probs, indices = probabilities.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    return probs, classes

def print_prediction(image_path, model):
    probs, classes = predict(image_path, model, topk=3)
    
    print(f"File: {image_path}")
    for i in range(3):
        class_name = cat_to_name.get(classes[i], "Unknown")
        print(f"Rank {i+1}: {class_name} (ID: {classes[i]}), Probability: {probs[i]:.2f}")

if model:
    print_prediction(image_path, model)