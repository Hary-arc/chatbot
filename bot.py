import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents, model, and other necessary files
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Function to clean up the sentence and tokenize it
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to create bag of words for prediction
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class (intent) of the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Function to get the response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Streamlit app interface
def run_chatbot():
    st.title("Healthcare Chatbot")
    st.write("Enter your symptoms, and the chatbot will search for a relevant diagnosis.")

    user_input = st.text_input("Describe your symptoms:")

    if st.button("Submit"):
        if user_input:
            # Predict the class and get response
            intents_list = predict_class(user_input)
            response = get_response(intents_list, intents)
            
            # Display the bot's response
            st.write(f"Bot Response: {response}")
        else:
            st.write("Please enter your symptoms to get a response.")

# Run the chatbot
if __name__ == "__main__":
    run_chatbot()
