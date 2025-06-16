import numpy as np
from tensorflow.keras.models import load_model
from keras.config import enable_unsafe_deserialization

# Allow unsafe deserialization for Lambda layers (trusted source)
enable_unsafe_deserialization()

# Load the trained ATCNet model
model_path = r"C:\\Users\\hp\ATC-NET2\\Results\\Train-2\\subject-2.keras"
model = load_model(model_path)

# Class-to-movement mapping
class_to_movement = {
    0: "left",   # Left Arm
    1: "right",  # Right Arm
    2: "down",   # Both Feet
    3: "up"      # Tongue
}

def predict_movement(eeg_input):
    """
    Predicts movement direction based on EEG input using the ATCNet model.
    
    Parameters:
        eeg_input (np.ndarray): EEG input of shape (14, 1125) or (1, 14, 1125).
        
    Returns:
        int: Predicted class (0 to 3)
        str: Corresponding movement direction
    """
    if eeg_input.shape == (14, 1125):
        eeg_input = np.expand_dims(eeg_input, axis=0)  # Add batch dimension → (1, 14, 1125)

    if eeg_input.shape == (1, 14, 1125):
        eeg_input = np.expand_dims(eeg_input, axis=1)  # Add channel dim → (1, 1, 14, 1125)

    if eeg_input.shape != (1, 1, 14, 1125):
        raise ValueError(f"Expected input shape (1, 1, 14, 1125), but got {eeg_input.shape}")

    predictions = model.predict(eeg_input)
    predicted_class = np.argmax(predictions, axis=1)[0]
    movement = class_to_movement[predicted_class]
    
    return predicted_class, movement

# Simulated EEG input (replace with actual EEG data)
sample_input = np.random.randn(14, 1125)
predicted_class, movement = predict_movement(sample_input)
print(f"Predicted Class: {predicted_class}, Movement: {movement}")
