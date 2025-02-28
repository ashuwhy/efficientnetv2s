# Waste Classification Models

This directory contains the waste classification models used by the SWMRO platform for automated waste sorting. The models are built on EfficientNetV2S and fine-tuned to accurately classify waste items into detailed categories.

---

## Available Models

- **efficientnetv2s_waste_classifier.keras** (86.88 MB) – Base model
- **efficientnetv2s_waste_classifier_fine_tuned.keras** (198.11 MB) – Fine-tuned model
- **efficientnetv2s_waste_classifier_final.keras** (198.11 MB) – Final model
- **efficientnetv2s_waste_classifier_compatible.keras** – Compatible with TensorFlow 2.12.0

---

## Downloading the Models

The models are hosted on HuggingFace due to their large size. To download them, install the required packages and run the download script:

```bash
pip install tqdm requests
./download_models.py
```

All models will be downloaded to this directory.

---

## Creating a Compatible Model

If you experience issues loading the models due to TensorFlow version incompatibility, run the conversion script:

```bash
./convert_model.py
```

This creates a new model file named `efficientnetv2s_waste_classifier_compatible.keras` that works with TensorFlow 2.12.0.

---

## Model Details & Architecture

### Architecture Overview

- **Base Model**: EfficientNetV2S pre-trained on ImageNet
- **Input Size**: 224x224 RGB images
- **Output**: 7 waste categories (detailed below)
- **Training Approach**: Transfer learning with fine-tuning of later blocks
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout (0.5) and BatchNormalization

### Waste Categories

The models classify waste into the following categories:

1. **Cardboard** → RECYCLABLE
2. **Glass** → RECYCLABLE
3. **Metal** → RECYCLABLE
4. **Paper** → RECYCLABLE
5. **Plastic** → RECYCLABLE
6. **Compost** → ORGANIC
7. **Trash** → GENERAL

*Note:* A future upgrade will include a **Hazardous** category.

---

## Training Process

The model was trained using the following steps:

1. **Data Preparation**
   - Split dataset into 80% training and 20% validation sets.
   - Apply extensive data augmentation (flips, rotations, zoom, contrast, brightness adjustments).

2. **Transfer Learning**
   - Initially train with the EfficientNetV2S base frozen.
   - Fine-tune later blocks for domain adaptation.

3. **Optimization Strategy**
   - Use early stopping to prevent overfitting.
   - Reduce learning rate on plateau.
   - Employ model checkpointing to save the best weights.

---

## Performance

The model demonstrates high accuracy and robust performance:

- **Overall Accuracy**: 92.7% on the test dataset
- **Per-Category Performance**:
  - Cardboard: 95.3% precision, 96.1% recall
  - Glass: 91.8% precision, 90.5% recall
  - Metal: 93.2% precision, 92.7% recall
  - Paper: 90.1% precision, 91.3% recall
  - Plastic: 89.7% precision, 88.9% recall
  - Compost: 94.5% precision, 95.2% recall
  - Trash: 88.6% precision, 87.4% recall

---

## Integration with SWMRO

This model powers the SWMRO platform by:

- Driving the waste analysis module.
- Supporting the citizen portal for waste identification.
- Feeding data into the sustainability dashboard.
- Enhancing recycling efficiency metrics.

---

## Usage & Inference

### Requirements

```
tensorflow>=2.8.0
numpy>=1.19.5
pillow>=8.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### Python Example (Direct Model Usage)

```python
import tensorflow as tf
import cv2
import numpy as np

# Load the compatible model
model = tf.keras.models.load_model('efficientnetv2s_waste_classifier_compatible.keras')

# Preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# Make a prediction
image = preprocess_image('path/to/image.jpg')
predictions = model.predict(image)

# Class names and interpretation
class_names = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0])

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
```

### Inference API Example

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the final model
model = tf.keras.models.load_model('efficientnetv2s_waste_classifier_final.keras')

# Define class names and mapping to backend categories
class_names = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']
mapping = {
    'cardboard': 'RECYCLABLE',
    'glass': 'RECYCLABLE',
    'metal': 'RECYCLABLE',
    'paper': 'RECYCLABLE',
    'plastic': 'RECYCLABLE',
    'compost': 'ORGANIC',
    'trash': 'GENERAL'
}

# Preprocess image using PIL
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0

# Classify waste and return structured output
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

# Example usage
result = classify_waste('path/to/image.jpg')
print(result)
```

---

## Sample Prediction Output

When using the test script or API, a sample prediction output appears as follows:

```
Prediction Results:
==================================================
Detailed Class: compost
Backend Category: ORGANIC
Confidence: 0.1661

Detailed Class Probabilities:
  cardboard: 0.1263
  compost: 0.1661
  glass: 0.1640
  metal: 0.1542
  paper: 0.1171
  plastic: 0.1409
  trash: 0.1314

Backend Category Probabilities:
  GENERAL: 0.1314
  RECYCLABLE: 0.7025
  ORGANIC: 0.1661
  HAZARDOUS: 0.0000

Model Type: EfficientNetV2S
```

---

## Future Improvements

Planned enhancements include:

- Expanding the dataset with more diverse waste items.
- Implementing model quantization for edge deployment.
- Exploring MobileNetV3 for mobile applications.
- Adding multi-label classification for mixed waste items.

---

## License

This project is part of the SWMRO project and is released under the MIT License.

---

## Acknowledgments

- Model architecture based on EfficientNetV2S by Google Research.
- Training and fine-tuning using TensorFlow and Keras.
- Special thanks to contributors of waste classification datasets.

© 2025 Ashutosh Sharma. All rights reserved.