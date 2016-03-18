"""PyAudio example: Record a few seconds of audio and save to a queue for later processing."""

import pyaudio
import wave
import StringIO
import Queue
import thread
import StringIO
from time import  sleep
import time
import math
from array import array
from sys import byteorder


STOP_FLAG = False

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "../audio/mic_input/test_output.wav"
RECORD_FLAG = True
THRESHOLD = 500


input_queue = Queue.Queue()
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    #print "Input volume at", max(snd_data)
    return max(snd_data) < THRESHOLD


def record_4_sec():
    while RECORD_FLAG == True:
        '''TODO: ADD STRINGIO'''
        p = pyaudio.PyAudio()
        #wav_buffer = StringIO.StringIO(buffer)
        #handle = wave.open('../audio/sample/input.wav', 'wb')
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK)
                snd_data = array('h', data)
                if byteorder == 'big':
                    snd_data.byteswap()
                #if not is_silent(snd_data):
                frames.append(data)
            except:
                pass



        if frames:
            print "writing out frames!"
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            input_queue.put([WAVE_OUTPUT_FILENAME, time.time()])
            wf.close()
            stream.stop_stream()
            stream.close()
            p.terminate()


        '''
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

        '''

def chop_input(wav_queue):
    while True:
        if not input_queue.empty():
            print "Chop Worker waking up...\n"
            dequeued_item = input_queue.get()
            handle = wave.open(dequeued_item[0], 'rb')
            frame_rate = handle.getframerate()
            n_frames = handle.getnframes()
            window_size = 2 * frame_rate
            num_secs = int(math.ceil(n_frames/frame_rate))

            snippet_list = []

            print "Slicing Audio file..."
            #for i in xrange(num_secs):
            for i in xrange(38):
                '''TODO: USE STRINGIO'''
                #wav_buffer = StringIO.StringIO(buffer)
                handle2 = '../audio/mic_input/'+str(i)+'snippet.wav'
                snippet = wave.open(handle2 ,'wb')
                snippet.setnchannels(2)
                snippet.setsampwidth(handle.getsampwidth())
                snippet.setframerate(frame_rate)
                snippet.writeframes(handle.readframes(window_size))
                try:
                    handle.setpos(handle.tell() - int(1.8 * frame_rate))
                    wav_queue.put([handle2, dequeued_item[1]])
                except:
                    pass
                snippet.close()

            handle.close()
        else:
            #print "Chop Worker waiting...\n"
            sleep(.2)

def run(wav_queue):
    print "running recorder...\n"

    #  chopper =Thread(target=chop_input, args = (wav_queue,))
    #  chopper.daemon = True
     #
    #  recorder = thread(target=record_8_sec)
    #  recorder.daemon = True
     #
    #  chopper.start()
    #  recorder.start()
    # try:
    thread.start_new_thread(record_4_sec, ())
    thread.start_new_thread(chop_input, (wav_queue,))
    # except:
    #     print "recording thread failed!!!"
