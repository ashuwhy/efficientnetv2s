# Waste Classification Model

## Model Files
Due to their large size, the model files are hosted externally:

- [efficientnetv2s_waste_classifier.keras](https://huggingface.co/ashuwhy/efficientnetv2s/resolve/main/efficientnetv2s_waste_classifier.keras) (86.88 MB) - Base model
- [efficientnetv2s_waste_classifier_fine_tuned.keras](https://huggingface.co/ashuwhy/efficientnetv2s/resolve/main/efficientnetv2s_waste_classifier_fine_tuned.keras) (198.11 MB) - Fine-tuned model
- [efficientnetv2s_waste_classifier_final.keras](https://huggingface.co/ashuwhy/efficientnetv2s/resolve/main/efficientnetv2s_waste_classifier_final.keras) (198.11 MB) - Final saved model

Please download these files and place them in this directory before using the model.

## Overview
This repository contains a deep learning model for waste classification, a critical component of the Smart Waste Management and Recycling Optimization Platform (SWMRO). The model is designed to classify waste items into 7 categories using EfficientNetV2S architecture, optimized for performance on Mac M2 chips.

![Waste Classification](https://img.shields.io/badge/AI-Waste%20Classification-brightgreen)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![EfficientNetV2S](https://img.shields.io/badge/Model-EfficientNetV2S-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-High%20Precision-success)

## Purpose
The waste classification model serves as the AI backbone for automated waste sorting in the SWMRO platform. It enables:

- **Accurate Waste Categorization**: Identifies waste items across 7 different categories
- **Recycling Optimization**: Improves recycling efficiency through precise classification
- **Environmental Impact**: Reduces landfill waste by ensuring proper waste sorting
- **Integration with SWMRO**: Powers the platform's waste analysis capabilities

## Model Architecture

### Technical Specifications
- **Base Model**: EfficientNetV2S (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels (RGB)
- **Classes**: 7 waste categories
- **Training Approach**: Transfer learning with fine-tuning
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout (0.5) and BatchNormalization

### Waste Categories
The model classifies waste into the following categories:
1. **Cardboard** → RECYCLABLE
2. **Glass** → RECYCLABLE
3. **Metal** → RECYCLABLE
4. **Paper** → RECYCLABLE
5. **Plastic** → RECYCLABLE
6. **Compost** → ORGANIC
7. **Trash** → GENERAL
8. **Hazardous** → HAZARDOUS [not used in the current model but used in swmro, this is just for future upgrade]

## Training Process

The model was trained using a comprehensive approach:

1. **Data Preparation**: 
   - Dataset split into training (80%) and validation (20%) sets
   - Extensive data augmentation (flips, rotations, zoom, contrast, brightness)

2. **Transfer Learning**:
   - Initial training with frozen EfficientNetV2S base
   - Fine-tuning of later blocks for domain adaptation

3. **Optimization Strategy**:
   - Early stopping to prevent overfitting
   - Learning rate reduction on plateau
   - Model checkpointing to save best weights

## Performance

The model achieves high accuracy in waste classification tasks, making it suitable for real-world applications. Performance metrics include:
- High classification accuracy across all waste categories
- Robust performance on varied waste item appearances
- Efficient inference time suitable for real-time applications

## Integration with SWMRO

This model integrates with the SWMRO platform to:
- Power the waste analysis module
- Support the citizen portal for waste identification
- Provide data for the sustainability dashboard
- Enhance recycling efficiency metrics

## Usage

### Requirements
```
tensorflow>=2.8.0
numpy>=1.19.5
pillow>=8.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### Inference Example
```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('efficientnetv2s_waste_classifier_final.keras')

# Class names
class_names = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Mapping to backend categories
mapping = {
    'cardboard': 'RECYCLABLE',
    'glass': 'RECYCLABLE',
    'metal': 'RECYCLABLE',
    'paper': 'RECYCLABLE',
    'plastic': 'RECYCLABLE',
    'compost': 'ORGANIC',
    'trash': 'GENERAL'
}

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0

# Classify waste
def classify_waste(image_path):
    preprocessed_img = preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    backend_category = mapping[predicted_class]
    
    return {
        'detailed_class': predicted_class,
        'backend_category': backend_category,
        'confidence': float(confidence)
    }
```

## Future Improvements

Planned enhancements for the model include:
- Expanding the dataset with more diverse waste items
- Implementing model quantization for edge deployment
- Exploring MobileNetV3 for mobile applications
- Adding multi-label classification for mixed waste

## License

This project is part of the SWMRO project and is available under the MIT License.

## Acknowledgments

- The model architecture is based on EfficientNetV2S by Google Research
- Training utilized TensorFlow and Keras frameworks
- Special thanks to contributors of waste classification datasets

© 2025 Ashutosh Sharma. All rights reserved.
