from headset.emotiv_lsl.emotiv_epoc_x import EmotivEpocX

def start_device(SRATE=128):
    emotiv_epoc_x = EmotivEpocX()
    emotiv_epoc_x.main_loop(SRATE)

if __name__ == "__main__":
    start_device()
