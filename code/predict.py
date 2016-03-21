import warnings
warnings.filterwarnings("ignore")

from time import sleep
import time
import numpy as np
import pandas as pd

def predict(fingerprint_queue,prediction_queue, model):
    while True:
        if not fingerprint_queue.empty():
            #print "Predictor Worker waking up...\n"
            fingerprint = fingerprint_queue.get()
            if fingerprint == 'SILENT_TRACK':
                prediction_queue.put('SILENT_TRACK')
            else:
                try:
                    X = fingerprint[0].reshape(1, -1)
                    prediction = model.predict(X)
                    #print "PREDICTION MADE AT: ", prediction
                    timestamp = time.time()
                    #print "TIME ELAPSED: ", timestamp - fingerprint[1]
                    prediction_queue.put([prediction, fingerprint[1]])
                except:
                    print "Couldn't make prediction on file"
                    pass



        else:
            #print "Predictor worker waiting....\n"
            sleep(.2)
