# Calcified-Polyp-Identification

This project was completed as part of my Bachelor Thesis in B.E. Computer Science Engineering in 2020. The objective was to develop a machine learning pipeline capable of identifying calcified polyps in ultrasound images using feature extraction and convolutional neural networks.

Due to medical data privacy regulations, the original dataset cannot be shared publicly.

# Image Classification Using Convolutional Neural Networks (CNN)

## Motivation
The inspiration for this project, Calcification of Polyp Identification, comes from the need to improve medical diagnostics. Polyps can vary in texture, with calcification being a key indicator of potential health concerns. Traditional ultrasound scans can detect polyps but do not analyze their texture effectively. By leveraging deep learning and image processing techniques, this project aims to automate the classification of polyps based on their calcification, assisting in early diagnosis and better medical decision-making.

## Abstract
This project focuses on binary classification of images using a Convolutional Neural Network (CNN). The model is trained to distinguish between two classes based on features extracted through multiple convolutional layers. The training process incorporates image augmentation techniques to enhance generalization and prevent overfitting. 

The entire workflow is divided into the following key phases:
1. **Image Preprocessing & Augmentation**: Images are resized and transformed using rotation, width/height shifts, shearing, zooming, and horizontal flipping.
2. **Dataset Preparation**: The images are loaded from directories into batches using the `ImageDataGenerator` class.
3. **CNN Model Creation**: A sequential CNN model is constructed with multiple convolutional, activation, pooling, flattening, and dense layers.
4. **Model Compilation**: The model is compiled using binary cross-entropy as the loss function and RMSprop as the optimizer.
5. **Training & Validation**: The model is trained using the augmented dataset, and validation is performed on a separate set of images.
6. **Performance Evaluation**: The validation accuracy and loss metrics are analyzed to assess model performance.
7. **Model Saving**: The trained model weights are saved for future use.

## Features
- **Data Augmentation**: Enhances dataset variability to improve generalization.
- **CNN-based Feature Extraction**: Uses convolutional layers to extract meaningful features from images.
- **Binary Classification**: Employs a sigmoid activation function for distinguishing between two classes.
- **Training with Keras**: Utilizes Keras data generators for model training with real-time augmentation.
- **Model Persistence**: Saves trained weights for further analysis and use.

## Dataset Description
The dataset consisted of anonymized ultrasound images classified into two categories:
- Calcified polyps
- Non-calcified polyps

All images were preprocessed, augmented and the dataset was split into training and validation sets to evaluate model generalization performance.

Due to medical privacy and ethical restrictions, the dataset cannot be publicly shared.

## Dataset Structure
The images should be organized in the following format:
```
project_root/
│── train/
│   ├── class_1/
│   ├── class_2/
│── validation/
│   ├── class_1/
│   ├── class_2/
```
- `train/`: Contains training images categorized into subfolders by class.
- `validation/`: Contains validation images organized similarly.

## Installation & Setup
1. Install the required dependencies:
   ```bash
   pip install tensorflow keras numpy matplotlib
   ```
2. Place your dataset inside the `train/` and `validation/` directories following the structure above.
3. Run the training script to train the model:
   ```bash
   python train.py
   ```

## Model Architecture
The CNN consists of:
- **Convolutional Layers**: Three Conv2D layers with ReLU activation.
- **Pooling Layers**: MaxPooling2D to reduce spatial dimensions.
- **Fully Connected Layers**: Dense layers with dropout to prevent overfitting.
- **Final Output Layer**: A single neuron with sigmoid activation for binary classification.

## Training the Model
The model is compiled with:
- **Loss Function**: Binary Crossentropy
- **Optimizer**: RMSprop
- **Metric**: Accuracy

It is trained using `fit_generator` with real-time data augmentation:
```python
model.fit_generator(
    train_generator,
    steps_per_epoch=13 // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=160 // batch_size
)
```

## Saving the Model
After training, the model weights are saved as:
```python
model.save_weights('50_epochs.h5')
```

## Conclusion
The results showed a clear improvement in model performance as training epochs increased. With 10 training epochs, the model achieved an accuracy of 66.67%. Increasing the training duration to 100 epochs improved accuracy to 75%, while extending training to 250 epochs resulted in 100% classification accuracy on the available test dataset. This indicates that sufficient training iterations are critical for enabling the model to learn distinguishing patterns effectively.

The project highlights the importance of feature extraction, data transformation, and iterative model training in improving classification performance. It also demonstrates my ability to work with complex medical imaging data, apply analytical techniques, and evaluate model performance using quantitative metrics.

| Training Epochs | Accuracy |
| --------------- | -------- |
| 10              | 66.67%   |
| 100             | 75%      |
| 250             | 100%     |


## Future Improvements
- Implement **early stopping** to prevent overfitting.
- Use **Adam optimizer** instead of RMSprop for better performance.
- Add **more convolutional layers** to improve feature extraction.
- Save the **entire model**, not just weights, for easier reusability.
