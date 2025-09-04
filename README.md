# Brain Tumor Detection using Deep Learning

This project uses a deep learning model to classify brain tumors from MRI scans. The model is built with Keras and TensorFlow, leveraging the VGG16 architecture through transfer learning for high accuracy.
***

## 🧐 Overview
Early and accurate diagnosis of brain tumors is crucial for effective treatment. This project automates the classification of brain tumors using a Convolutional Neural Network (CNN). By fine-tuning a pre-trained VGG16 model, we can achieve robust performance even with a limited dataset. The model can classify MRI scans into several categories, such as Glioma, Meningioma, Pituitary, and No Tumor.

***

## Features
* **Multi-Class Classification**: Identifies multiple types of brain tumors from MRI images.
* **Transfer Learning**: Utilizes the powerful VGG16 model, pre-trained on the vast ImageNet dataset, as a feature extractor.
* **Fine-Tuning**: The last three convolutional layers of VGG16 are unfrozen and retrained to adapt the model specifically for the brain MRI dataset.
* **Regularization**: Implements **Dropout** layers to reduce overfitting and improve the model's ability to generalize to new, unseen data.

***

## Model Architecture
The model is constructed sequentially, starting with the VGG16 base and adding custom classification layers on top.

1.  **Input Layer**: The model accepts images of size $128 \times 128$ with 3 color channels (RGB).
2.  **Base Model (VGG16)**: We use the VGG16 model without its top classification layer (`include_top=False`). Most of its layers are **frozen** (`layer.trainable = False`) to retain the learned ImageNet weights.
    * **Fine-Tuning**: The last three convolutional layers (`block5_conv1`, `block5_conv2`, `block5_conv3`) are made **trainable** to allow the model to learn features specific to MRI scans.
3.  **Flatten Layer**: This layer converts the 2D feature maps from the VGG16 base into a 1D vector.
4.  **Dropout Layer**: A dropout rate of `0.3` is applied to randomly deactivate 30% of neurons during training, preventing co-adaptation of neurons.
5.  **Dense Hidden Layer**: A fully connected layer with **128 neurons** and a **ReLU** (`relu`) activation function to learn complex patterns from the features.
6.  **Second Dropout Layer**: Another dropout layer with a rate of `0.2` is applied for further regularization.
7.  **Output Layer**: The final `Dense` layer uses a **softmax** activation function to output a probability score for each tumor class. The number of neurons equals the number of classes in the dataset.

### Training Configuration
* **Optimizer**: **Adam** with a learning rate of $0.0001$.
* **Loss Function**: **`sparse_categorical_crossentropy`**, which is ideal for multi-class classification problems where the labels are provided as integers.
* **Metrics**: **`sparse_categorical_accuracy`** is monitored to evaluate the model's performance during training.

***

## 📊 Dataset
The model was trained on the **Brain Tumor MRI Dataset**, which contains images for four distinct classes:
1.  Glioma
2.  Meningioma
3.  Pituitary Tumor
4.  No Tumor

All images are preprocessed by resizing them to $128 \times 128$ pixels before being fed into the model.

***

## ⚙️ Installation
To get this project running on your local machine, follow these steps.
  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/brain-tumor-detection.git](https://github.com/your-username/brain-tumor-detection.git)
    cd brain-tumor-detection
    ```
***

## 🚀 Usage
To classify a new brain MRI scan, you can use the following script.

1.  Place your MRI scan image in a known directory.
2.  Run the prediction script. Make sure to update the model path and image path.
