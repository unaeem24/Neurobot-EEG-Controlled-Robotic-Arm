import multiprocessing
import time
from colorama import Fore, Style, init

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from headset.run import start_device
from headset.record import record

from vrepcom.atcnetpred_csv import predict_movement
from vrepcom.atcnetpred_csv import execute_movement

 
model_path = r'C:\Users\hp\ATC-NET2\Results\subject-2.keras'

# Initialize colorama


class_to_movement = {
    0: "left", 1: "right", 2: "down", 3: "up"
}

def print_instructions():
    print(Fore.YELLOW + "Imagine Moving Following Body Parts To Control The Arm")
    print(Fore.YELLOW + Style.BRIGHT + "Left Arm" + Fore.YELLOW + Style.NORMAL + " if you want to move " + Fore.YELLOW + Style.BRIGHT + "Left")
    print(Fore.YELLOW + Style.BRIGHT + "Right Arm"+ Fore.YELLOW + Style.NORMAL +" if you want to move " + Fore.YELLOW + Style.BRIGHT + "Right")
    #print(Fore.YELLOW + Style.BRIGHT + "Feet"+ Fore.YELLOW + Style.NORMAL + " if you want to move " + Fore.YELLOW + Style.BRIGHT + "Down")
    #print(Fore.YELLOW + Style.BRIGHT + "Tongue" + Fore.YELLOW + Style.NORMAL + " if you want to move " + Fore.YELLOW + Style.BRIGHT+ "Up")

def record_and_predict():
    print(Fore.GREEN + "Recording Starting in 5 seconds")
    time.sleep(5)
    print(Fore.GREEN + Style.BRIGHT + "Recording Starting")
    data = record(128,9)
    print(Fore.GREEN + Style.BRIGHT + "Recording stopped")
    pred_class = predict_movement(data, model_path)
    
    return pred_class

def move_arm(sim, joint1, joint2, outcome):
    execute_movement(sim, joint1, joint2, outcome)
    time.sleep(2)


def run_headset():
    headset_process = multiprocessing.Process(target=start_device, args=(128,))
    headset_process.daemon = True
    headset_process.start()

def main():
    headset_process = run_headset()
    # while True:
    print_instructions()
    pred_class = record_and_predict()
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    sim.startSimulation()
    time.sleep(0.5)
    joint1 = sim.getObject('/joint1')
    joint2 = sim.getObject('/joint2')
    print("Predicted class: ", pred_class[0].item())
    class_index = pred_class[0].item()
    move_arm(sim, joint1, joint2, class_index)
    # outcome = class_to_movement[pred_class]
    # #close the headset process
    # move_arm(sim, joint1, joint2, outcome)
        
        

if __name__ == "__main__":
   
    main()
