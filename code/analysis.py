'''Running tests for optimal classification algorithms'''

#Grab Sample Audio Snippets and add noise to them build test data out of noise files
handle = wave.open('Adele_Solo.wav')

snippets = chop_song(handle, "noisy")

noisy_snippets = add_noise(snippets)




#Apply Transformation 1  to input signal
fingerprint = transform_multiple(noisy_snippets)



#Apply Ceptstra Tranformation to signal
fingerprint2 = cepstra_transform(noisy_snippets)


#Run Predictions on both Transformations (Random Forest, LogisticRegression, Decision Tree, SVM, Naïve Bayes)
model = getModel('models/rf.pkl')

prediction = model.predict(fingerprint)
#Create Analysis Report
print 'prediction with random forest:' prediction






##TEST 2: Adding  Algorithmic Noise to model

#Run Predictions


#Fit Model with K Nearest Neighbors

#Run Predictions

#Write Analysis Report



#TEST 3:  Try with microphone recorded input
