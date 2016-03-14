"""PyAudio example: Record a few seconds of audio and save to a queue for later processing."""

import pyaudio
import wave
import StringIO
import Queue
import thread
import StringIO
from time import sleep
import math

STOP_FLAG = False

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 8
WAVE_OUTPUT_FILENAME = "../audio/test_output.wav"

p = pyaudio.PyAudio()

input_queue = Queue.Queue()

def record_8_sec():
    while True:
        '''TODO: ADD STRINGIO'''
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
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        input_queue.put(WAVE_OUTPUT_FILENAME)
        wf.close()


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
            handle = wave.open(input_queue.get(), 'rb')
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
                handle2 = '../audio/sample/'+str(i)+'snippet.wav'
                snippet = wave.open(handle2 ,'wb')
                snippet.setnchannels(2)
                snippet.setsampwidth(handle.getsampwidth())
                snippet.setframerate(frame_rate)
                snippet.writeframes(handle.readframes(window_size))
                handle.setpos(handle.tell() - int(1.8 * frame_rate))
                snippet.close()
                wav_queue.put(handle2)
            handle.close()
        else:
            print "Chop Worker waiting...\n"
            sleep(1)

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
    thread.start_new_thread(record_8_sec, ())
    thread.start_new_thread(chop_input, (wav_queue,))
    # except:
    #     print "recording thread failed!!!"
