from time import sleep

def predict(fingerprint_queue,prediction_queue, model):
    while True:
        if not fingerprint_queue.empty():
            fingerprint = fingerprint_queue.get()
            prediction = model.predict_proba(fingerprint)
            print "PREDICTION MADE AT: ", prediction
            prediction_queue.put(prediction)
        else:
            print "Predictor worker waiting....\n"
            sleep(1)
