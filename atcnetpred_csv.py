import numpy as np
import pandas as pd
import math
import time
import os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from tensorflow.keras.models import load_model
from keras.config import enable_unsafe_deserialization
import tensorflow as tf
# Allow unsafe deserialization
enable_unsafe_deserialization()

# Folder containing all 9 model files
# model_folder = r'C:\Users\hp\ATC-NET2\Results'
# model_files = [f for f in os.listdir(model_folder) if f.endswith('.keras')]
# model_files.sort()  # Optional: ensure consistent order

# # CSV file path
# csv_path = r'C:\Users\hp\ATC-NET2\signal_file\signal-up movementmz.csv'

# Class mapping
class_to_movement = {
    0: "left", 1: "right", 2: "down", 3: "up"
}

# Load EEG data from CSV
def load_eeg_from_csv(csv_path):
    df = pd.read_csv(csv_path, header=None, skiprows=1)  # Skip first row if needed
    data = df.values.astype(np.float32)
    if data.shape[1] != 14:
        raise ValueError("Expected 14 EEG channels per sample (14 columns).")
    if data.shape[0] < 1125:
        raise ValueError("Not enough rows in the CSV. At least 1125 needed.")
    trimmed = data[:1125, :].T  # Shape becomes (14, 1125)
    return trimmed

# Predict function
def predict_movement(eeg_input, model_path):
    # Load the ATCNet model
    model = tf.keras.models.load_model(model_path)
    
    # Prepare the input (expected shape: [batch_size, channels, height, width])
    eeg_input = np.expand_dims(np.expand_dims(eeg_input, axis=0), axis=0)  # (1, 1, 14, 1125)
    
    # Predict
    predictions = model.predict(eeg_input)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return predicted_class, class_to_movement[predicted_class], predictions
# Execute robot movement

def execute_movement(sim, joint1, joint2, prediction: int):
    print("Executing movement for prediction:", prediction)
    if prediction == 0:
        sim.setJointTargetPosition(joint1, math.radians(-90))
    elif prediction == 1:
        sim.setJointTargetPosition(joint1, math.radians(90))
    elif prediction == 2:
        sim.setJointTargetPosition(joint2, math.radians(-90))
    elif prediction == 3:
        sim.setJointTargetPosition(joint2, math.radians(90))
    else:
        print("Unknown prediction:", prediction)

# # try:
#     print("Loading EEG data...")
#     eeg_data = load_eeg_from_csv(csv_path)
#     print("Data shape:", eeg_data.shape)

#     # Connect to CoppeliaSim
#     client = RemoteAPIClient()
#     sim = client.getObject('sim')
#     sim.startSimulation()
#     time.sleep(0.5)
#     joint1 = sim.getObject('/joint1')
#     joint2 = sim.getObject('/joint2')
#     print("Connected to CoppeliaSim")

#     print("\nRunning predictions with 9 models...\n")

#     for idx, model_file in enumerate(model_files):
#         model_path = os.path.join(model_folder, model_file)
#         print(f"\n[{idx+1}/7] Loading model: {model_file}")
#         model = load_model(model_path)
#         class_idx, movement, probs = predict_movement(eeg_data, model)
#         confidence = np.max(probs)
#         print(f"Model {idx+1}: Class {class_idx} → {movement}, Confidence: {confidence:.2%}")

#     # Optionally execute movement based on the last model
#     print("\nExecuting movement using last model's prediction...")
#     execute_movement(sim, joint1, joint2, class_idx)
#     time.sleep(2)
#     sim.stopSimulation()

# except Exception as e:
#     print(f"\nFatal error: {str(e)}")
