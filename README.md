# SENTIMENT-ANALYSIS

This project is an AI-powered Sentiment Analysis Chatbot built using Python and NLP libraries.
It analyzes the emotional tone of user inputs in real-time and responds with appropriate messages.
The chatbot uses a hybrid approach combining VADER (from NLTK) and TextBlob sentiment analysis techniques for improved accuracy.

Features

 Text preprocessing (tokenization, stopword removal, lemmatization)
 Sentiment detection (Positive / Negative / Neutral)
 Hybrid scoring using VADER + TextBlob
 Intelligent chatbot responses based on mood
 Conversation logging in JSON format (conversation_logs.json)
 Works with the Sentiment140 dataset for large-scale sentiment testing

Tech Stack

Language: Python 

Libraries:

NLTK (VADER Sentiment, Tokenization, Stopwords)

TextBlob (Polarity scoring)

Pandas & NumPy (Data handling)

re (Text cleaning with regex)

JSON (Conversation logging)


Example Chat
Launching AI Chatbot with Sentiment Analysis...
Type 'exit' to stop the chat.

You: I am feeling great today!
Bot: 😊 I'm happy that you're feeling positive! (confidence: 0.85)

You: I am really tired and sad.
Bot: 😔 I'm sorry you're feeling that way. (confidence: 0.77)

You: exit
Bot: 👋 Goodbye! Have a nice day.

🧠 How It Works

Preprocessing: Cleans text (removes URLs, mentions, punctuation).

Tokenization & Lemmatization: Breaks sentences into root words.

Dual Sentiment Analysis:

VADER → Lexicon-based scoring

TextBlob → Polarity-based scoring

Weighted Average: Combines both methods (60% VADER + 40% TextBlob).

Response Generation: Generates mood-aware chatbot replies.

Logging: Saves all conversations and scores to conversation_logs.json.




This project is licensed under the MIT License – feel free to use, modify, and share with credit.
