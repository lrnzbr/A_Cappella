import warnings
warnings.filterwarnings("ignore")

import pyaudio
import wave
import sys
from time import sleep
from collections import Counter
import thread
import time
import math
from train import convert_to_timestamp


CHUNK = 1024
MISMATCH_WARNING_COUNT = 0
STOPPED_FLAG = True
LAST_PREDICTION = 0
CONGESTED_QUEUE_COUNT = 0

def convert_mark_to_seconds(mark):
    parts = mark.split(":")
    minutes = int(parts[0])
    seconds = int(parts[1])
    return minutes*60 + seconds

def play_at_prediction(mark, time_delay=0):
    print "playing at ", convert_to_timestamp(mark), "with time delay of", time.time() - time_delay
    p = pyaudio.PyAudio()
    wf = wave.open('../audio/originals/Adele_Instrumental.wav', 'rb')
    framerate = wf.getframerate()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=framerate,
                    output=True)
    framesize = wf.getsampwidth() * wf.getnchannels()
    #wf.setpos(mark*wf.getframerate() + int(wf.getframerate()*(time.time()-time_delay)))
    delta = int(time.time() - time_delay)
    if delta < 10:
            wf.setpos(framerate*int(mark + (delta)))
    else:
        wf.setpos(framerate*int(mark))


    data = wf.readframes(CHUNK)

    while data != '':
        if STOPPED_FLAG == False:
            stream.write(data)
            data = wf.readframes(CHUNK)
        else:
            # print "MUSIC STOPPED!, stopped flag = ", STOPPED_FLAG
            break

    stream.stop_stream()
    stream.close()
    #print "FILE CLOSED!"

    p.terminate()

def pull_predictions(prediction_queue):
    global LAST_PREDICTION
    global STOPPED_FLAG
    global MISMATCH_WARNING_COUNT
    global CONGESTED_QUEUE_COUNT
    mode = False
    while True:
        if prediction_queue.qsize() >= 10:
            #print "Play Worker waking up...\n"
            predictions_list = list()
            time_delay = list()
            for i in xrange(10):
                prediction = prediction_queue.get()
                predictions_list.append(prediction)
            if "SILENT_TRACK" not in predictions_list:
                time_delay.append(prediction[1])
                mode_finder = Counter(item[0][0] for item in predictions_list)
                mode = mode_finder.most_common(1)
                print "PREDICTED SOLOIST AT:", mode[0][0]
                mode = convert_mark_to_seconds(mode[0][0])

            elif prediction == "SILENT_TRACK":
                    if LAST_PREDICTION != 0:
                        print "SILENCE IN TRACK, PAUSEING...."
                    STOPPED_FLAG = True
                    prediction_queue.queue.clear()
                    LAST_PREDICTION = 0
                    mode = False

            #CASE 0: MUSIC HAS YET TO PLAY:
            if LAST_PREDICTION == 0:
                # print "CASE 0:"
                if mode:
                    STOPPED_FLAG = False
                    thread.start_new_thread(play_at_prediction, (mode, time_delay[-1],))
                    LAST_PREDICTION = mode
                    MISMATCH_WARNING_COUNT = 0
            # CASE 1: MUSIC IS ALREADY PLAYING AND PREDICTIONS ARE STILL ON TRACK
            elif mode > LAST_PREDICTION and mode < LAST_PREDICTION + 10:
                #print "CASE 1: MUSIC IS ALREADY PLAYING AND PREDICTIONS ARE STILL ON TRACK"
                LAST_PREDICTION = mode
            #CASE 2: MUSIC HAS STOPPED DUE TO MISMATCH. START AGAIN AT NEW PREDICTION AND RESET FLAGS
            elif STOPPED_FLAG:
                #print "#CASE 2: MUSIC HAS STOPPED DUE TO MISMATCH. START AGAIN AT NEW PREDICTION AND RESET FLAGS"
                STOPPED_FLAG = False
                play_at_prediction(mode, time_delay=time.time() - time_delay[-1])
                LAST_PREDICTION = mode
                MISMATCH_WARNING_COUNT = 0
            #CASE 3: MUSIC IS PLAYING BUT MODE DOESN'T SEEM TO BE MATCHING UP, RAISE WARNING COUNT
            elif mode < LAST_PREDICTION or mode > LAST_PREDICTION + 10:
                #print "#CASE 3: MUSIC IS PLAYING BUT MODE DOESN'T SEEM TO BE MATCHING UP, RAISE WARNING COUNT"
                MISMATCH_WARNING_COUNT += 1

            if MISMATCH_WARNING_COUNT > 5:
                print "MISMATCH!!!!!....RECALCULATING......"
                STOPPED_FLAG = True
                LAST_PREDICTION = 0
                MISMATCH_WARNING_COUNT = 0

        else:
            # #IF A SILENT TRACK IS SLOWING DOWN THE QUEUE, CLEAR THE QUEUE AND START OVER
            # if LAST_PREDICTION != 0:
            #     CONGESTED_QUEUE_COUNT +=1
            #     if CONGESTED_QUEUE_COUNT >= 250:
            #         print "CONGESTED QUEUE!!"
            #         STOPPED_FLAG = True
            #         LAST_PREDICTION = 0
            #         prediction_queue.queue.clear()
            #         CONGESTED_QUEUE_COUNT = 0
            #         break
            sleep(.2)
