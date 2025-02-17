## **🚀 Overview**
# **AI Image Classifier - Final Project**
This repository contains the final project for the AI Programming with Python course. The project focuses on building an **Image Classifier** using pre-trained models  **VGG16** and **DenseNet121** to predict flower species from images.
script to demonstrate how to train a convolutional network and use it for prediction (~8000 flower images)

## **🌟 Features**



* Utilizes **PyTorch** for model training, loading and predictions
* Supports **VGG16** and **DenseNet121** architectures
* Option to run on **GPU** if available
* Processes and normalizes images for accurate predictions
* Displays top-3 prediction results with probabilities


## **📁 Project Structure**

.

├── predict.py                  # Main script to make predictions

├── train.py                  # Main script to train the model and create pth checkpoints

├── cat_to_name.json            # Mapping of category labels to flower names

├── flowers/                    # Folder containing validation images data source: 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'

├── model_checkpoint_vgg16.pth  # VGG16 trained model checkpoint download link: https://huggingface.co/MaraKo/model_checkpoint_vgg16/resolve/main/model_checkpoint_vgg16.pth?download=true

├── model_checkpoint_densenet121.pth  # DenseNet121 trained model checkpoint 
https://huggingface.co/MaraKo/model_checkpoint_vgg16/resolve/main/model_checkpoint_vgg16.pth?download=true

└── README.md                   # Project documentation


## **🔧 Installation**
python 3.12.5
torch, torch vision, pillow, numpy, pandas, matplotlib, 
flower data that has been used to train two versions of the model: link above
model checkpoints for two models link above 
keep all files in 1 folder



## **🔢 Usage**


### **Command Line Arguments train.py:**

python train.py ./flowers --batch_size 64 --model densenet121 --learning_rate 0.001 --epochs 3 --hidden_units 512 --checkpoint_dir ./ --device cuda

* `data_dir`: Path to the data
* `batch size`: size of the batch
* model : model you want to use for training (Vgg16 and densenet121)
* learning rate: to test (0,001 was not too bad
* epochs: number to test (3 worked well)
* hidden_units: 512 This argument defines the number of neurons (units) in the hidden layers of the classifier section of your model.
* checkpoint_dir: folder where *pth file shall be saved
* `--gpu`: Tesla GPU needed 90 minutes to process the training, CPU is not recommended

### **Command Line Arguments predict.py:**

Run the prediction script with the desired model and image:
python predict.py ./model_checkpoint_densenet121.pth ./your path to image/your_image.jpg
Or with GPU:
python predict.py ./model_checkpoint_vgg16.pth ./your path to image/your_image.jpg --gpu



* `checkpoint`: Path to the trained model (`.pth` file)
* `image`: Path to the image file for prediction
* `--gpu`: Optional flag to use GPU if available


## **📊 Example Output**

Checkpoint successfully loaded from: ./model_checkpoint_vgg16.pth

File: ./your_path/your_image.jpg

Rank 1: Purple Coneflower (ID: 70), Probability: 0.85

Rank 2: Bee Balm (ID: 34), Probability: 0.10

Rank 3: Cardinal Flower (ID: 12), Probability: 0.05


## **🎓 Learning Outcomes**

* Gained hands-on experience with **PyTorch**
* Learned how to load and fine-tune **pre-trained models**
* Applied **image preprocessing** techniques for model input
* Enhanced understanding of **command-line argument parsing** using `argparse`

## **🚜 Future Improvements**

* Add support for more pre-trained models
* Implement a web interface for easier image uploads and predictions
* Improve model accuracy with additional training data

## **👁️ License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The dataset used in this project is sourced from Udacity and is subject to their terms of use. Please refer to Udacity's [Terms of Use](https://www.udacity.com/legal/terms-of-service) for more information on the usage rights of their content.
Copyright (c) 2025 Mara Koeppen


---

📢 **Contact \
** For any questions, feel free to reach out at your.email@example.com
