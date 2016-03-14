from record import run
from transform import transform
from predict import predict
from play import pull_predictions
import thread
import Queue
from train import getModel
from time import sleep


wav_queue = Queue.Queue()
fft_queue = Queue.Queue()
prediction_queue = Queue.Queue()


# recording_thread = Thread(target=run, args=(wav_queue,))
# tranformation_thread = Thread(target=transform, args=(wav_queue, fft_queue))
# prediction_thread = Thread(target=predict, args=(fft_queue, prediction_queue))
# playing_thread = Thread(target=pull_predictions, args = (prediction_queue,))
#
#
# recording_thread.daemon = True
# tranformation_thread.daemon = True
# prediction_thread.daemon = True
# playing_thread.daemon = True
#
# playing_thread.start()
# prediction_thread.start()
# tranformation_thread.start()
# recording_thread.start()

model = model = getModel('../models/rf.pkl')
try:
   thread.start_new_thread( transform, (wav_queue, fft_queue, ) )
   thread.start_new_thread(predict, (fft_queue, prediction_queue, model))
   thread.start_new_thread(pull_predictions, (prediction_queue,))
   thread.start_new_thread( run, (wav_queue, ) )

except:
   print "Error: unable to start threads"
while 1:
    pass
