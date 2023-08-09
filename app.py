import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as ps



# ...
# Rest of your imports and code

# Create an instance of PorterStemmer
stemmer = ps()

def transform_sms(Sms):
    Sms = Sms.lower()
    words = nltk.word_tokenize(Sms)
    word_list1 = []
    word_list2 = []
    word_list3 = []

    for word in words:
        if word.isalnum():
            word_list1.append(word)

    for word in word_list1:
        if word not in stopwords.words("english"):
            word_list2.append(word)

    for word in word_list2:
        word_list3.append(stemmer.stem(word))  # Use the stem() method

    return " ".join(word_list3)


import pickle

# Load a pickled object from a file
with open("vectorizers1.pkl", "rb") as file:
    tfidf = pickle.load(file)
model = pickle.load(open('model.pkl','rb'))



st.title("Email/message SPAM classifier")
input_sms = st.text_area("Enter the message")
if st.button("predict"):
    # 1 preprocess

    transformed_sms = transform_sms(input_sms)

    vector_input=tfidf.transform([transformed_sms])


    result=model.predict(vector_input)[0]

    if result==1:
        st.header("spam")
    else:
        st.header("not spam")
