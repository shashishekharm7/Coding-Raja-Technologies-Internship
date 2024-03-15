import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of conversations
conversations = [
    {"text": "Hello", "intent": "greeting", "response": "Hi there! How can I assist you today?"},
    {"text": "How can I help you?", "intent": "question", "response": "I'm sorry, I'm still learning. Could you please ask another question?"},
    {"text": "What are your hours?", "intent": "question", "response": "Our business hours are from 9 AM to 5 PM, Monday to Friday."},
    # Add more sample conversations here
]

# Preprocess text data
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Preprocess conversations
corpus = [preprocess_text(conv['text']) for conv in conversations]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Intent recognition function
def recognize_intent(user_input):
    user_input = preprocess_text(user_input)
    user_input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vector, X)
    max_similarity_index = np.argmax(similarities)
    return conversations[max_similarity_index]['intent']

# Response generation function
def generate_response(intent):
    for conv in conversations:
        if conv['intent'] == intent:
            return conv['response']
    return "I'm sorry, I didn't understand that."

# Main function for chatbot
def chatbot():
    print("Chatbot: Hi there! How can I assist you today?")
    while True:
        user_input = input("User: ")
        intent = recognize_intent(user_input)
        response = generate_response(intent)
        print("Chatbot:", response)

# Run the chatbot
if _name_ == "_main_":
    chatbot()