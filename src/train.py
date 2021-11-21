
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score

pd.options.mode.chained_assignment = None  # Set the dummpy pandas warning off.

dataset = pd.read_csv('dataset.csv')

dataset = dataset.sample(frac=1).reset_index(drop=True) # Suffling dataset for homogeneous dataset distribution

dataset.sort_values("Body", inplace = True) # Short by 'Body' column

dataset = dataset.drop(columns="B") # Drop unnecessary 'B' column from dataset

dataset.drop_duplicates(subset ="Body",keep = False, inplace = True) # Drop duplicated rows in the dataset by 'Body' subset.

def optimization(dataset):
    dataset = dataset.dropna()

    stop_words = set(stopwords.words('turkish'))
    punctuals = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(punctuals)

    for ind in dataset.index:
        body = dataset['Body'][ind] # Get body from row with 'ind' index
        body = body.lower() # Converting all of text to lower body because of preventing unrelated case sensitive text effect
        body = re.sub(r'http\S+', '', body) # removing URL's from text
        body = re.sub('\[[^]]*\]', '', body) # removing html tags from text
        body = (" ").join([word for word in body.split() if not word in stop_words]) # Clear stopwords from text
        body = "".join([char for char in body if not char in punctuals]) # Clear punctuals by replacing them with whitespace (\0)
        dataset['Body'][ind] = body # Replace optimized text 
    return dataset


dataset = optimization(dataset)

comments = dataset['Body']
labels =  dataset['Label']

tfIdf = TfidfVectorizer( binary=False, ngram_range=(1,3))
tfIdf.fit(comments)

comments_machine = dataset[dataset['Label']==0]
comments_human = dataset[dataset['Label']==1]


comments_machine_vectors = tfIdf.transform(comments_machine['Body'].tolist())
comments_human_vectors = tfIdf.transform(comments_human['Body'].tolist())

# printing one of vector for inspecting how does a vector look like 
print("="*5) 
print(comments_machine_vectors[0])
print("="*5)

# Now, convert train data to vectors with tfIdf
comments = tfIdf.transform(comments)    

# comments = X | labels = Y
x_train, x_test, y_train, y_test = train_test_split(comments, labels, test_size=0.2, random_state=0)

# Create instance of LogisticRegression class
lojistikRegresyon = LogisticRegression() 

# Train LogisticRegression object with .fit()
lojistikRegresyon.fit(x_train,y_train)

# Predict randomly selected x_test test vectors with our new trained LogisticRegression model
y_predicted = lojistikRegresyon.predict(x_test)

# Print confusion matrix of binomial LogisticRegression model
print("confusion matrix: ", confusion_matrix(y_test,y_predicted))

# Print classification report of binomial LogisticRegression model
print("accuracy score: ",accuracy_score(y_test,y_predicted))

#Save trained LogisticRegression model as binary
pickle.dump(lojistikRegresyon, open("bin/trained_lr_model.bin", 'wb'))
print("Logistic regression model has been trained successfully and saved to disk")

#Save trained vectorizer model as binary
pickle.dump(tfIdf, open("bin/vector_model.bin", 'wb'))
print("Tf-Idf vectorizer model has been saved to disk.")

