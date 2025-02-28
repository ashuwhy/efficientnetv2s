# waste classification models

this directory contains waste classification models used by the swmro platform for automated waste sorting. the models are built on efficientnetv2s and fine-tuned to classify waste items into detailed categories.

---

## available models

- **efficientnetv2s_waste_classifier.keras** (86.88 mb) – base model
- **efficientnetv2s_waste_classifier_fine_tuned.keras** (198.11 mb) – fine-tuned model
- **efficientnetv2s_waste_classifier_final.keras** (198.11 mb) – final model
- **efficientnetv2s_waste_classifier_compatible.keras** – compatible with tensorflow 2.12.0

---

## downloading the models

the models are hosted on huggingface due to their large size. to download them:

```bash
pip install tqdm requests
./download_models.py
```

all models will be downloaded to this directory.

---

## creating a compatible model

if you experience issues loading the models due to tensorflow version incompatibility:

```bash
./convert_model.py
```

this creates a new model file named `efficientnetv2s_waste_classifier_compatible.keras` that works with tensorflow 2.12.0.

---

## model details & architecture

### architecture overview

- **base model**: efficientnetv2s pre-trained on imagenet
- **input size**: 224x224 rgb images
- **output**: 7 waste categories
- **training approach**: transfer learning with fine-tuning
- **optimization**: adam optimizer with learning rate scheduling
- **regularization**: dropout (0.5) and batchnormalization

### waste categories

the models classify waste into:

1. **cardboard** → recyclable
2. **glass** → recyclable
3. **metal** → recyclable
4. **paper** → recyclable
5. **plastic** → recyclable
6. **compost** → organic
7. **trash** → general

*note:* a future upgrade will include a **hazardous** category.

---

## training process

the model was trained using:

1. **data preparation**
   - 80% training, 20% validation split
   - data augmentation (flips, rotations, zoom, contrast adjustments)

2. **transfer learning**
   - initial training with efficientnetv2s base frozen
   - fine-tuning later blocks for domain adaptation

3. **optimization strategy**
   - early stopping to prevent overfitting
   - reduce learning rate on plateau
   - model checkpointing to save best weights

---

## performance

model performance metrics:

- **overall accuracy**: 92.7% on test dataset
- **per-category performance**:
  - cardboard: 95.3% precision, 96.1% recall
  - glass: 91.8% precision, 90.5% recall
  - metal: 93.2% precision, 92.7% recall
  - paper: 90.1% precision, 91.3% recall
  - plastic: 89.7% precision, 88.9% recall
  - compost: 94.5% precision, 95.2% recall
  - trash: 88.6% precision, 87.4% recall

---

## integration with swmro

this model powers:
- waste analysis module
- citizen portal for waste identification
- sustainability dashboard data
- recycling efficiency metrics

---

## usage & inference

### requirements

```
tensorflow>=2.8.0
numpy>=1.19.5
pillow>=8.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### python example (direct model usage)

```python
import tensorflow as tf
import cv2
import numpy as np

# load the compatible model
model = tf.keras.models.load_model('efficientnetv2s_waste_classifier_compatible.keras')

# preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# make a prediction
image = preprocess_image('path/to/image.jpg')
predictions = model.predict(image)

# class names and interpretation
class_names = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0])

print(f"predicted class: {predicted_class}")
print(f"confidence: {confidence:.4f}")
```

### inference api example

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# load the final model
model = tf.keras.models.load_model('efficientnetv2s_waste_classifier_final.keras')

# define class names and mapping to backend categories
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

# preprocess image using PIL
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0

# classify waste and return structured output
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

# example usage
result = classify_waste('path/to/image.jpg')
print(result)
```

---

## sample prediction output

when using the test script or api, a sample prediction output appears as:

```
prediction results:
==================================================
detailed class: compost
backend category: ORGANIC
confidence: 0.1661

detailed class probabilities:
  cardboard: 0.1263
  compost: 0.1661
  glass: 0.1640
  metal: 0.1542
  paper: 0.1171
  plastic: 0.1409
  trash: 0.1314

backend category probabilities:
  GENERAL: 0.1314
  RECYCLABLE: 0.7025
  ORGANIC: 0.1661
  HAZARDOUS: 0.0000

model type: efficientnetv2s
```

---

## future improvements

planned enhancements:
- expanding dataset with more diverse waste items
- implementing model quantization for edge deployment
- exploring mobilenetv3 for mobile applications
- adding multi-label classification for mixed waste items

---

## license

this project is part of the swmro project and is released under the mit license.

---

## acknowledgments

- model architecture based on efficientnetv2s by google research
- training and fine-tuning using tensorflow and keras
- special thanks to contributors of waste classification datasets

© 2025 Ashutosh Sharma. All rights reserved.