import pickle
from sklearn.svm import SVC
import numpy as np
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words('turkish'))
punctuals = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
stop_words.update(punctuals)

# This function is same the optimization function in 'train.py' but specified for one string.
def optimization(text):
    text = text.lower() # Converting all of text to lower body because of preventing unrelated case sensitive text effect
    text = re.sub(r'http\S+', '', text) # removing URL's from text
    text = re.sub('\[[^]]*\]', '', text) # removing html tags from text
    text = (" ").join([word for word in text.split() if not word in stop_words]) # Clear stopwords from text
    text = "".join([char for char in text if not char in punctuals]) # Clear punctuals by replacing them with whitespace (\0)
    return text

text2Predict  = optimization(input("Enter a comment to be classified by ML:"))

# Load trained TfIdf model from disk
tfIdf = pickle.load(open("bin/vector_model.bin", 'rb'))

# Vectorize input text using pre-trained TfIdf model
text2Predict_vector = tfIdf.transform([text2Predict])

# Load SVM model from disk
supportVectorMachine = pickle.load(open("bin/trained_svm_model.bin", 'rb'))

# predict with model
predictionResult = supportVectorMachine.predict(text2Predict_vector)

if(predictionResult == 0):
    print("This comment is classified as machine-wrote")
else:
    print("This comment is classified as human-wrote")