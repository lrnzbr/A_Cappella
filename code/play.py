"""PyAudio Example: Play a WAVE file."""

import pyaudio
import wave
import sys
from time import sleep
from collections import Counter
import thread
import time


CHUNK = 1024
MISMATCH_WARNING_COUNT = 0
STOPPED_FLAG = True
LAST_PREDICTION = 0




def play_at_prediction(mark, time_delay=0):
    print "playing...."
    p = pyaudio.PyAudio()
    wf = wave.open('../audio/originals/Adele_Instrumental.wav', 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    framesize = wf.getsampwidth() * wf.getnchannels()
    #wf.setpos(mark*wf.getframerate() + int(wf.getframerate()*(time.time()-time_delay)))
    wf.setpos(wf.getframerate()*int(mark + (time.time()-time_delay)))
    data = wf.readframes(CHUNK)

    while data != '' and STOPPED_FLAG == False:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    print "FILE CLOSED!"

    p.terminate()

def pull_predictions(prediction_queue):
    global LAST_PREDICTION
    global STOPPED_FLAG
    global MISMATCH_WARNING_COUNT

    while True:
        if prediction_queue.qsize() >= 10:
            print "Play Worker waking up...\n"
            predictions_list = list()
            time_delay = list()
            for i in xrange(10):
                prediction = prediction_queue.get()
                predictions_list.append(prediction[0][0])
                time_delay.append(prediction[1])

            mode_finder = Counter(predictions_list)
            mode = mode_finder.most_common(1)
            #CASE 0: MUSIC HAS YET TO PLAY:
            if LAST_PREDICTION == 0:
                print "CASE 0:"
                STOPPED_FLAG = False
                thread.start_new_thread(play_at_prediction, (mode[0][0], time_delay[-1],))
                LAST_PREDICTION = mode
                MISMATCH_WARNING_COUNT = 0
            # CASE 1: MUSIC IS ALREADY PLAYING AND PREDICTIONS ARE STILL ON TRACK
            elif mode > LAST_PREDICTION and mode < LAST_PREDICTION + 10:
                print "CASE 1: MUSIC IS ALREADY PLAYING AND PREDICTIONS ARE STILL ON TRACK"
                LAST_PREDICTION = mode
            #CASE 2: MUSIC HAS STOPPED DUE TO MISMATCH. START AGAIN AT NEW PREDICTION AND RESET FLAGS
            elif STOPPED_FLAG:
                print "#CASE 2: MUSIC HAS STOPPED DUE TO MISMATCH. START AGAIN AT NEW PREDICTION AND RESET FLAGS"
                STOPPED_FLAG = False
                play_at_prediction(mode, delta=time.time() - time_delay[-1])
                LAST_PREDICTION = mode
                MISMATCH_WARNING_COUNT = 0
            #CASE 3: MUSIC IS PLAYING BUT MODE DOESN'T SEEM TO BE MATCHING UP, RAISE WARNING COUNT
            elif mode < LAST_PREDICTION or mode > LAST_PREDICTION + 10:
                print "#CASE 3: MUSIC IS PLAYING BUT MODE DOESN'T SEEM TO BE MATCHING UP, RAISE WARNING COUNT"
                MISMATCH_WARNING_COUNT += 1
            if MISMATCH_WARNING_COUNT > 3:
                print "MISMATCH!!!!!....RECALCULATING......"
                STOPPED_FLAG = True
        else:
            #print "play worker waiting...\n"
            sleep(.2)
