# Waste-and-Garbage-Detection
Waste and Garbage Classifier
Introduction
This project is a Waste and Garbage Classifier that aims to classify images of waste and garbage into different categories. The classifier is based on a custom ResNet model trained on a dataset of waste and garbage images. The model is able to predict the category of waste or garbage depicted in an input image.

Dataset
The dataset used for training the classifier was obtained from Kaggle (insert link to the dataset here). It consists of images belonging to different categories of waste and garbage, such as plastic, paper, metal, organic, etc.

Model Architecture
The model architecture used for this classifier is a custom ResNet model with a pre-trained ResNet-18 backbone. The last fully connected layer of the pre-trained ResNet-18 is replaced with a new fully connected layer to match the number of output classes.

How to Use
Clone the repository to your local machine.
Install the required libraries and dependencies using pip install -r requirements.txt.
Download the dataset and place it in the data directory.
Train the model using the train.py script. You can modify the hyperparameters and other configurations in the script.
After training, the model will be saved as custom_resnet_model.pth in the models directory.
Use the predict.py script to make predictions on new images using the trained model. Place the images you want to predict in the predict directory.
The predictions will be displayed along with the predicted class and confidence score.

Performance
The model's performance can be evaluated on a separate test dataset. It achieves an accuracy of approximately 61% on the test set.

Next Steps
Fine-tune the model by experimenting with different architectures and hyperparameters to improve performance.
Deploy the model as a web application or API for easy access and use.
Collect more data and expand the dataset to include a wider range of waste and garbage categories.
Implement data augmentation techniques to increase the diversity of the training dataset.
Explore techniques to handle imbalanced classes, if present in the dataset.

Acknowledgements
The dataset used for training this model was obtained from Kaggle (https://www.kaggle.com/code/aadhavvignesh/pytorch-garbage-classification-95-accuracy). We would like to thank the creators of the dataset for providing this valuable resource.
We also thank the developers and contributors of PyTorch and torchvision for their excellent libraries and tools, which made this project possible.
