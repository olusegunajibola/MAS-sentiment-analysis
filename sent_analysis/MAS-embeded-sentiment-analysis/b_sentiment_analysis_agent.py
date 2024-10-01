import torch
import transformers
import numpy as np
import pickle
import pika
import json

import warnings
warnings.filterwarnings('ignore')

class SentimentAnalysisAgent:
    def __init__(self, distilbert_path='D:\Data\PyCharmProjects\MAS-sentiment-analysis\sent_analysis\models/DB_n_LR/DB.pth',
                 logistic_model_path='D:\Data\PyCharmProjects\MAS-sentiment-analysis\sent_analysis\models/DB_n_LR/logistic_regression_model.pkl'):
        # Load the DistilBERT model and tokenizer
        self.model_class, self.tokenizer_class, self.pretrained_weights = (
            transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')

        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.distilbert = self.model_class.from_pretrained(self.pretrained_weights)

        # Load the trained DistilBERT model state
        self.distilbert.load_state_dict(torch.load(distilbert_path, map_location=torch.device('cpu')))
        self.distilbert.eval()  # Set the model to evaluation mode

        # Load the trained Logistic Regression model
        with open(logistic_model_path, 'rb') as file:
            self.lr_model = pickle.load(file)

    def preprocess_text(self, text):
        # Tokenize the input text
        tokenized = self.tokenizer.encode(text, add_special_tokens=True)

        # Padding
        max_len = 512  # Default for DistilBERT is 512 tokens
        padded_token_embeddings = np.array(tokenized + [0] * (max_len - len(tokenized)))

        # Attention mask
        attention_mask = np.where(padded_token_embeddings != 0, 1, 0)

        return padded_token_embeddings, attention_mask

    def get_bert_features(self, text):
        # Preprocess the input text
        padded_token_embeddings, attention_mask = self.preprocess_text(text)

        # Convert inputs to torch tensors
        input_ids = torch.tensor([padded_token_embeddings])
        attention_mask = torch.tensor([attention_mask])

        # Extract last hidden states from DistilBERT
        with torch.no_grad():
            outputs = self.distilbert(input_ids, attention_mask=attention_mask)

        # Extract the [CLS] token's hidden state (first token in sequence)
        features = outputs[0][:, 0, :].numpy()

        return features

    def predict_sentiment(self, text):
        # Extract features from DistilBERT
        features = self.get_bert_features(text)

        # Predict sentiment using the Logistic Regression model
        sentiment = self.lr_model.predict(features)

        return sentiment

    # def send_message_to_rabbitmq(self, text, sentiment):
    #     # Create connection to RabbitMQ
    #     connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    #     channel = connection.channel()
    #
    #     # Declare queue
    #     channel.queue_declare(queue='sentiment')
    #
    #     # Prepare the message
    #     message = {'text': text, 'sentiment': sentiment[0]}
    #     message = json.dumps(message)
    #
    #     # Send the message
    #     channel.basic_publish(exchange='', routing_key='sentiment', body=message)
    #     print(f'Sent sentiment: {sentiment[0]} for text: {text}')
    #
    #     # Close connection
    #     connection.close()

    # def send_message_to_rabbitmq(self, text, sentiment):
    #     # Convert the sentiment to a serializable format
    #     sentiment = sentiment[0] if isinstance(sentiment, np.ndarray) else sentiment
    #
    #     # Create connection to RabbitMQ
    #     connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    #     channel = connection.channel()
    #
    #     # Declare queue
    #     channel.queue_declare(queue='sentiment')
    #
    #     # Prepare the message
    #     message = {'text': text, 'sentiment': str(sentiment)}
    #     message = json.dumps(message)
    #
    #     # Send the message
    #     channel.basic_publish(exchange='', routing_key='sentiment', body=message)
    #     print(f'Sent sentiment: {sentiment}, for text: {text}')
    #
    #     # Close connection
    #     connection.close()

    def send_message_to_rabbitmq(self, text, sentiment):
        # Convert sentiment to an integer if it's a NumPy array or ensure it's in the correct format
        sentiment = int(sentiment[0]) if isinstance(sentiment, np.ndarray) else int(sentiment)

        # Create connection to RabbitMQ
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()

        # Declare queue
        channel.queue_declare(queue='sentiment')

        # Prepare the message
        message = {'text': text, 'sentiment': sentiment}
        message = json.dumps(message)

        # Send the message
        channel.basic_publish(exchange='', routing_key='sentiment', body=message)
        print(f'Sent sentiment: {sentiment} for text: {text}')

        # Close connection
        connection.close()

    def process_and_send(self, text):
        # Predict sentiment
        sentiment = self.predict_sentiment(text)

        # Send the sentiment to RabbitMQ
        self.send_message_to_rabbitmq(text, sentiment)
