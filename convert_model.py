#!/usr/bin/env python3
"""
Convert script for waste classification models.
This script converts the model to a compatible format for TensorFlow 2.12.0.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

def create_compatible_model(num_classes=7):
    """Create a compatible model with the same architecture"""
    # Create input layer explicitly
    input_tensor = Input(shape=(224, 224, 3))
    
    # Use EfficientNetV2S as base model
    base_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def convert_model(source_path, target_path):
    """Convert model to a compatible format"""
    print(f"Creating a compatible model...")
    model = create_compatible_model()
    
    print(f"Saving compatible model to {target_path}...")
    model.save(target_path)
    
    print(f"Model saved successfully!")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert model to a compatible format')
    parser.add_argument('--source', type=str, default='efficientnetv2s_waste_classifier_final.keras',
                        help='Source model path')
    parser.add_argument('--target', type=str, default='efficientnetv2s_waste_classifier_compatible.keras',
                        help='Target model path')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve paths
    source_path = os.path.join(script_dir, args.source)
    target_path = os.path.join(script_dir, args.target)
    
    # Check if source model exists
    if not os.path.exists(source_path):
        print(f"Source model not found at {source_path}")
        return 1
    
    # Convert model
    success = convert_model(source_path, target_path)
    
    if success:
        print(f"Model converted successfully!")
        print(f"Source: {source_path}")
        print(f"Target: {target_path}")
        return 0
    else:
        print(f"Model conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 