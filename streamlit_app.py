import streamlit as st
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import os

# Show title and description with a header
st.title("💬 Healthcare Assistant")
st.markdown("""
    This is a simple healthcare chatbot powered by machine learning. 
    Ask your health-related queries, and the chatbot will assist you with an answer.
    You can describe your symptoms or ask about general health issues.
""")
st.markdown("---")

# Custom CSS to enhance the visual appearance
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
        }
        .chat-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .bot-response {
            color: #0072B1;
            font-size: 16px;
            font-weight: 600;
            margin-top: 15px;
        }
        .user-input {
            background-color: #dbe6f0;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            margin-top: 10px;
        }
        .submit-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .submit-button:hover {
            background-color: #45a049;
        }
        .title-text {
            text-align: center;
            font-size: 30px;
            color: #0072B1;
            font-weight: bold;
        }
        .bot-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents, model, and other necessary files
def load_resources():
    try:
        intents = json.loads(open("intents.json").read())  # Load your intents file
        words = pickle.load(open('words.pkl', 'rb'))  # Load words
        classes = pickle.load(open('classes.pkl', 'rb'))  # Load classes
        model = load_model('chatbot_model.h5')  # Load trained chatbot model
        return intents, words, classes, model
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        raise SystemExit("Exiting due to resource loading failure.")

# Load resources
intents, words, classes, model = load_resources()

# Function to clean up the sentence and tokenize it
def clean_up_sentence(sentence):
    try:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words
    except Exception as e:
        st.error(f"Error processing the sentence: {str(e)}")
        return []

# Function to create bag of words for prediction
def bag_of_words(sentence):
    try:
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)
    except Exception as e:
        st.error(f"Error creating bag of words: {str(e)}")
        return []

# Function to predict the class (intent) of the sentence
def predict_class(sentence):
    try:
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
        return return_list
    except IndexError as e:
        st.error(f"Index error during prediction: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error predicting the class: {str(e)}")
        return []

# Function to get the response based on the predicted intent
def get_response(intents_list, intents_json):
    try:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    except IndexError as e:
        st.error(f"Index error in response selection: {str(e)}")
        return "Sorry, I couldn't find a relevant response."
    except KeyError as e:
        st.error(f"Key error while fetching response: {str(e)}")
        return "Sorry, there seems to be an issue processing your request."
    except Exception as e:
        st.error(f"Error getting the response: {str(e)}")
        return "Sorry, I encountered an error while processing your query."

# Save new data to the intents file for self-learning
def self_learn(user_input, response):
    try:
        new_intent = {
            "tag": "new_symptom",
            "patterns": [user_input],
            "responses": [response]
        }
        intents['intents'].append(new_intent)
        with open('intents.json', 'w') as f:
            json.dump(intents, f, indent=4)
        st.write("Thank you for teaching me! I've learned this new symptom and response.")
    except Exception as e:
        st.error(f"Error saving new data: {str(e)}")

# Streamlit app interface
def run_chatbot():
    st.markdown('<div class="title-text">Talk to Your Healthcare Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Create a container to hold the chatbot messages and inputs
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Display bot avatar and response
        st.image("https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg", caption="Bot Avatar", width=50, use_column_width=True)
        st.markdown("<p class='bot-response'>Hello! I'm your Healthcare Assistant. How can I assist you today?</p>", unsafe_allow_html=True)

        # User input box
        user_input = st.text_input("Describe your symptoms or ask a question:", key="user_input", placeholder="e.g., I have a headache")

        # Display the submit button
        if st.button("Submit", key="submit_button", help="Click to get the bot's response", use_container_width=True):
            if user_input:
                # Predict the class and get response
                intents_list = predict_class(user_input)
                if intents_list:
                    response = get_response(intents_list, intents)
                else:
                    response = "Sorry, I could not understand your symptoms. Would you like to teach me? (yes/no)"
                    self_learn_option = st.radio("Do you want to teach me?", options=["yes", "no"])
                    if self_learn_option == "yes":
                        new_response = st.text_input("Please provide the response for this symptom:")
                        if new_response:
                            self_learn(user_input, new_response)
                            response = "Thank you for teaching me!"
                    else:
                        response = "Okay, I will try to learn more later."
                
                # Display the bot's response
                st.markdown(f"<p class='bot-response'>{response}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='bot-response'>Please enter your symptoms or a question.</p>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Run the chatbot
if __name__ == "__main__":
    run_chatbot()
