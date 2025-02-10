# Transfer Learning with ResNet50 in TensorFlow/Keras

## Overview

This script demonstrates how to implement transfer learning using the pre-trained **ResNet50** model from TensorFlow/Keras. Transfer learning leverages the knowledge of an existing model trained on a large dataset to improve the performance of a new task with limited data. In this example, we customize ResNet50 for a new classification task.

## Code Breakdown

### 1. Import Necessary Libraries

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
```

- **TensorFlow**: The core library for building and training machine learning models.
- **ResNet50**: A pre-trained deep learning model available in Keras, trained on ImageNet.
- **Dense, Flatten**: Layers to build the custom classifier.
- **Model**: The Keras Model class to define the complete model architecture.

### 2. Load Pre-trained ResNet50 Model

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

- `weights='imagenet'`: Loads weights pre-trained on the ImageNet dataset.
- `include_top=False`: Excludes the top fully connected layers, allowing customization.
- `input_shape=(224, 224, 3)`: Specifies the input shape for the model.

### 3. Freeze Base Model Layers

```python
for layer in base_model.layers:
    layer.trainable = False
```

- **Freezing Layers**: Prevents the weights of the base model from being updated during training, ensuring that the pre-trained features are retained.

### 4. Add Custom Classifier Layers

```python
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='softmax')(x)
```

- **Flatten Layer**: Converts the 2D feature maps into a 1D feature vector.
- **Dense Layer (256 units)**: Adds a fully connected layer with ReLU activation.
- **Dense Layer (128 units)**: Adds another fully connected layer with Softmax activation.

**Note**: The final Dense layer uses a `softmax` activation function with 128 units. Ensure that the number of units matches the number of classes in your classification task. For binary classification, use 1 unit with a `sigmoid` activation function.

### 5. Define the New Model

```python
model = Model(inputs=base_model.input, outputs=x)
```

- **Model Definition**: Combines the base model and the custom classifier into a single model.

### 6. Compile the Model

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

- **Optimizer**: Uses the Adam optimizer for training.
- **Loss Function**: Uses categorical cross-entropy, suitable for multi-class classification tasks.
- **Metrics**: Tracks accuracy during training and evaluation.

### 7. Model Summary

```python
model.summary()
```

- **Model Summary**: Displays the architecture of the model, including the number of parameters in each layer.

## Additional Considerations

- **Data Preparation**: Ensure that your input data is preprocessed to match the expected input shape and scale. For image data, this often involves resizing images to `(224, 224)` and normalizing pixel values.

- **Training**: After compiling the model, proceed with training using your dataset. Monitor the training process to ensure that the model is learning effectively.

- **Evaluation**: After training, evaluate the model's performance on a separate validation or test dataset to assess its accuracy and generalization capability.

## References

For more detailed information on transfer learning and fine-tuning with TensorFlow/Keras, refer to the official TensorFlow tutorial:

- [Transfer Learning and Fine-Tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
