# run_eeg_control.py
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import time
import numpy as np

# atcnet_model.py
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained ATC-Net model once
model = load_model('path/to/your/atcnet_model.h5')  # <- change this

# Prediction function
def predict_eeg_action(eeg_input: np.ndarray) -> int:
    """
    Accepts: eeg_input with shape (1, channels, timepoints) or whatever your model expects
    Returns: prediction class as an int
    """
    prediction = model.predict(eeg_input)
    predicted_class = int(np.argmax(prediction))
    return predicted_class


# ========== STEP 1: Get EEG Input ==========
def get_eeg_sample():
    """
    Simulates loading or generating a single EEG sample.
    Replace this with actual preprocessing pipeline.
    """
    return np.random.randn(1, 22, 1125)  # Example shape: (1, channels, timepoints)

# ========== STEP 2: Map prediction to robot control ==========
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

# ========== STEP 3: Main Control Logic ==========
def run_eeg_control():
    # Get input
    eeg_input = get_eeg_sample()

    # Predict using ATC-Net
    prediction = predict_eeg_action(eeg_input)
    print("Predicted Class:", prediction)

    # Connect to CoppeliaSim
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    sim.startSimulation()
    time.sleep(0.5)

    # Get joints
    joint1 = sim.getObject('/joint1')
    joint2 = sim.getObject('/joint2')

    # Execute action based on prediction
    execute_movement(sim, joint1, joint2, prediction)

    time.sleep(2)
    sim.stopSimulation()

# ========== RUN ==========
if __name__ == '__main__':
    run_eeg_control()
