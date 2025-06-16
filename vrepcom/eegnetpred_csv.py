import numpy as np
import pandas as pd
import math
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from tensorflow.keras.models import load_model
from keras.config import enable_unsafe_deserialization
import os

# Allow unsafe deserialization
enable_unsafe_deserialization()

# Model and file paths
model_path = r'C:\Users\hp\ATC-NET2\Results\subject-3.keras'
csv_path = r'C:\Users\hp\ATC-NET2\signal_file\signal-up movementmz.csv'

# Verify files exist
print(f"Checking model file exists: {os.path.exists(model_path)}")
print(f"Checking CSV file exists: {os.path.exists(csv_path)}")

# Load model
try:
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit()

# Class mapping
class_to_movement = {
    0: "left", 1: "right", 2: "down", 3: "up"
}

def pad_eeg(eeg, target_length=1125):
    padded = np.zeros((14, target_length), dtype=np.float32)
    padded[:, :eeg.shape[1]] = eeg
    return padded

def load_eeg_from_csv(csv_path):
    """Load and validate EEG data, excluding the first row."""
    try:
        df = pd.read_csv(csv_path, header=None, skiprows=1)
        print(f"CSV loaded. Shape after skipping first row: {df.shape}")
        
        eeg_data = df.values.astype(np.float32)

        if eeg_data.shape == (384, 14):
            eeg_data = eeg_data.T
        elif eeg_data.shape != (14, 384):
            raise ValueError(f"Unexpected shape {eeg_data.shape}. Need (14, 384) or (384, 14)")
        
        print(f"Data reshaped to: {eeg_data.shape}")
        return eeg_data

    except Exception as e:
        print(f"Error in load_eeg_from_csv: {str(e)}")
        raise

def predict_movement(eeg_input):
    """Make prediction with shape validation and padding"""
    print(f"Original input shape: {eeg_input.shape}")
    
    # Pad to (14, 1125)
    eeg_input = pad_eeg(eeg_input)

    # Reshape to (1, 1, 14, 1125)
    eeg_input = np.expand_dims(np.expand_dims(eeg_input, axis=0), axis=0)
    print(f"Padded and reshaped input shape: {eeg_input.shape}")
    
    predictions = model.predict(eeg_input)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class, class_to_movement[predicted_class], predictions

def execute_movement(sim, joint1, joint2, prediction: int):
    if prediction == 0:  # Left
        sim.setJointTargetPosition(joint1, math.radians(-45))
    elif prediction == 1:  # Right
        sim.setJointTargetPosition(joint1, math.radians(45))
    elif prediction == 2:  # Up
        sim.setJointTargetPosition(joint2, math.radians(-30))
    elif prediction == 3:  # Down
        sim.setJointTargetPosition(joint2, math.radians(30))
    elif prediction == 4:  # Rest
        sim.setJointTargetPosition(joint1, 0)
        sim.setJointTargetPosition(joint2, 0)
    else:
        print("Unknown prediction:", prediction)

# Main execution
try:
    print("\nLoading EEG data...")
    eeg_data = load_eeg_from_csv(csv_path)
    
    print("\nMaking prediction...")
    class_idx, movement, probs = predict_movement(eeg_data)
    
    print(f"\nPrediction Result:")
    print(f"Class: {class_idx}, Movement: {movement}")
    print(f"Confidence: {np.max(probs):.2%}")

    # execute movement in copppeliasim
    # Connect to CoppeliaSim
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    print("Connected to CoppeliaSim")
    sim.startSimulation()
    time.sleep(0.5)

     # Get joints
    joint1 = sim.getObject('/joint1')
    joint2 = sim.getObject('/joint2')

    print("Joints obtained")

    print("\nExecuting movement...")

    # Execute action based on prediction
    execute_movement(sim, joint1, joint2, class_idx)
    time.sleep(2)
    sim.stopSimulation()

except Exception as e:
    print(f"\nFatal error: {str(e)}")
