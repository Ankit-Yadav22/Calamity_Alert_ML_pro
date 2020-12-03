from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
print("hello world")
pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

import numpy as np
import spacy
 #Need to load the large model to get the vectors
import en_core_web_lg
nlp = en_core_web_lg.load()

text = "happy new year"
with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector])
Ypred = pickle_model.predict(doc_vectors)

print(Ypred)