import warnings
warnings.filterwarnings("ignore")
from record import run
from transform import transform
from predict import predict

from play import pull_predictions
import thread
import Queue
from train import getModel
from time import sleep






if __name__ == '__main__':
    print "Firing up A Cappella!"
    wav_queue = Queue.Queue()
    fft_queue = Queue.Queue()
    prediction_queue = Queue.Queue()



    model = getModel('../models/LogisticRegPeakAnalysis.pkl')

    try:
        thread.start_new_thread( transform, (wav_queue, fft_queue, ) )
        thread.start_new_thread(predict, (fft_queue, prediction_queue, model))
        thread.start_new_thread(pull_predictions, (prediction_queue,))
        thread.start_new_thread( run, (wav_queue, ) )
    except:
       print "Error: unable to start threads"
    while 1:
        pass
