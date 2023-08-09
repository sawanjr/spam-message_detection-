import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as ps

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pickled objects
with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)
model = pickle.load(open('model.pkl', 'rb'))

# Create an instance of PorterStemmer
stemmer = ps()

# Function to transform SMS
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

# Streamlit UI
st.title("Email/Message Spam Classifier")
st.markdown("This app predicts whether an email/message is spam or not spam.")
input_sms = st.text_area("Enter the message")
if st.button("Predict"):
    if not input_sms:
        st.warning("Please enter a message.")
    else:
        # Preprocess the input SMS
        transformed_sms = transform_sms(input_sms)

        # Vectorize and predict
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Display prediction
        if result == 1:
            st.markdown("**Prediction:** This message is **spam**.")
        else:
            st.markdown("**Prediction:** This message is **not spam**.")

# Example messages for user guidance
st.sidebar.header("Example Messages:")
st.sidebar.write("1. Congratulations! You've won a prize worth $1000.")
st.sidebar.write("2. Hi, how are you doing? Let's catch up sometime.")
st.sidebar.write("3. Urgent: Your account needs verification. Click the link to proceed.")
