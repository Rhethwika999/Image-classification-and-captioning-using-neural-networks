
# Image Classificaton and Captioning using neural networks 🌅

## Aim 🔍
The aim of this project is to develop a comprehensive system that can classify images into predefined categories and generate descriptive captions for the images.

## Objectives 💡
### Image Classification
- **Objective**: To accurately classify images into their respective categories using a deep learning model.
- **Built Using**: Convolutional Neural Networks (CNNs)

### Image Captioning 
- **Objective**: To generate meaningful captions for images using a combination of CNNs and Recurrent Neural Networks (RNNs).
- **Built Using**: CNNs for feature extraction and RNNs for sequence generation

## Dataset Details 📊
- **Image Classification Dataset**: This project will use a subset of images from Tiny ImageNet, which is a subset of the ImageNet dataset. Our subset of Tiny ImageNet contains 30 different categories, we will refer to it as TinyImageNet30. The training set has 450 resized images (64x64 pixels) for each category (13,500 images in total).We have used CIFAR10 dataset for transfer learning which is available from  torchvision.datasets.CIFAR10 (https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
  
- **Image Captioning Dataset**: This project utilizes a subset of the COCO (Common Objects in Context: https://cocodataset.org/#home) dataset for image caption generation. The complete COCO dataset comprises 330,000 images spanning 80 object categories, with each image accompanied by at least five textual reference captions. Our selected subset, referred to as COCO_5070, includes approximately 5,070 images, each annotated with five or more unique descriptions highlighting the prominent objects and activities within the images.

## Libraries and Packages Used 📦
- **TensorFlow/Keras**: For building and training the neural networks
- **NumPy**: For numerical computations
- **Pandas**: For data manipulation and analysis
- **Matplotlib/Seaborn**: For data visualization
- **OpenCV**: For image processing
- **NLTK**: For natural language processing tasks

## Methodology 📈
### Image Classification
1. **Data Preprocessing**: Load and preprocess the images (resizing, normalization).
2. **Model Architecture**: Design a CNN architecture suitable for the classification task.
3. **Training**: Train the model on the training dataset.
4. **Evaluation**: Evaluate the model on the validation/test dataset.
5. **Optimization**: Fine-tune the model to improve accuracy.

### Image Captioning
1. **Data Preprocessing**: Load images and captions, preprocess the text (tokenization, padding).
2. **Feature Extraction**: Use a pre-trained CNN to extract features from images.
3. **Model Architecture**: Design a combined CNN-RNN architecture for caption generation.
4. **Training**: Train the model using the image features and corresponding captions.
5. **Evaluation**: Evaluate the model using metrics like BLEU score.
6. **Optimization**: Fine-tune the model to improve caption quality.

## Implementation Steps 🤔
1. **Setup Environment**: Install necessary libraries and packages.
2. **Data Preparation**: Download and preprocess the datasets.
3. **Model Development**:
   - **Image Classification**: Build, train, and evaluate the CNN model.
   - **Image Captioning**: Build, train, and evaluate the CNN-RNN model.
4. **Testing**: Test the models on new images to verify their performance.
5. **Deployment**: Deploy the models for real-time image classification and captioning.


