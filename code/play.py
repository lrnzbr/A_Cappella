"""PyAudio Example: Play a WAVE file."""

import pyaudio
import wave
import sys
from time import sleep


CHUNK = 1024

wf = wave.open('../audio/Adele_Instrumental.wav', 'rb')


p = pyaudio.PyAudio()



def play_at_prediction(mark, delta=0):
    print "playing...."
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    wf.setpos(mark*wf.getframerate() + delta)
    data = wf.readframes(CHUNK)

    while data != '':
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()

def pull_predictions(prediction_queue):
    while True:
        if not prediction_queue.empty():
            prediction = prediction_queue.get()
            mark = prediction[0]
            play_at_prediction(mark, delta=0)
            print "PREDICTED MARK: ", mark
        else:
            print "play worker waiting...\n"
            sleep(1)
