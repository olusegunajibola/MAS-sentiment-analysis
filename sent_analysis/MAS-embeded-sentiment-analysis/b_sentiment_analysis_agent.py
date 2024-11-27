# import torch
# import transformers
# import numpy as np
# import pickle
# import json
# from a_connection import get_rabbitmq_connection
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# class SentimentAnalysisAgent:
#     def __init__(self, distilbert_path='D:/Data/PyCharmProjects/MAS-sentiment-analysis/sent_analysis/models/DB_n_LR/DB.pth',
#                  logistic_model_path='D:/Data/PyCharmProjects/MAS-sentiment-analysis/sent_analysis/models/DB_n_LR/logistic_regression_model.pkl'):
#         # Load the DistilBERT model and tokenizer
#         self.model_class, self.tokenizer_class, self.pretrained_weights = (
#             transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
#
#         self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
#         self.distilbert = self.model_class.from_pretrained(self.pretrained_weights)
#
#         # Load the trained DistilBERT model state
#         self.distilbert.load_state_dict(torch.load(distilbert_path, map_location=torch.device('cpu')))
#         self.distilbert.eval()  # Set the model to evaluation mode
#
#         # Load the trained Logistic Regression model
#         with open(logistic_model_path, 'rb') as file:
#             self.lr_model = pickle.load(file)
#
#     def preprocess_text(self, text):
#         # Tokenize the input text
#         tokenized = self.tokenizer.encode(text, add_special_tokens=True)
#
#         # Padding
#         max_len = 512  # Default for DistilBERT is 512 tokens
#         padded_token_embeddings = np.array(tokenized + [0] * (max_len - len(tokenized)))
#
#         # Attention mask
#         attention_mask = np.where(padded_token_embeddings != 0, 1, 0)
#
#         return padded_token_embeddings, attention_mask
#
#     def get_bert_features(self, text):
#         # Preprocess the input text
#         padded_token_embeddings, attention_mask = self.preprocess_text(text)
#
#         # Convert inputs to torch tensors
#         input_ids = torch.tensor([padded_token_embeddings])
#         attention_mask = torch.tensor([attention_mask])
#
#         # Extract last hidden states from DistilBERT
#         with torch.no_grad():
#             outputs = self.distilbert(input_ids, attention_mask=attention_mask)
#
#         # Extract the [CLS] token's hidden state (first token in sequence)
#         features = outputs[0][:, 0, :].numpy()
#
#         return features
#
#     def predict_sentiment(self, text):
#         # Extract features from DistilBERT
#         features = self.get_bert_features(text)
#
#         # Predict sentiment using the Logistic Regression model
#         sentiment = self.lr_model.predict(features)
#
#         print(f"Predicted sentiment for text '{text}': {sentiment[0]}")
#         return sentiment
#
#     def send_message_to_rabbitmq(self, text, sentiment):
#         # Convert sentiment to an integer if it's a NumPy array or ensure it's in the correct format
#         sentiment = int(sentiment[0]) if isinstance(sentiment, np.ndarray) else int(sentiment)
#
#         # Prepare the message
#         message = {'text': text, 'sentiment': sentiment}
#         message_json = json.dumps(message)
#
#         try:
#             # Create connection to RabbitMQ
#             connection = get_rabbitmq_connection()
#             channel = connection.channel()
#
#             # Declare a direct exchange
#             channel.exchange_declare(exchange='sentiment_exchange', exchange_type='direct')
#
#             # Determine the routing key based on sentiment
#             if sentiment == 0:
#                 routing_key = 'negative'
#             elif sentiment == 1:
#                 routing_key = 'neutral'
#             elif sentiment == 2:
#                 routing_key = 'positive'
#             else:
#                 print(f"Unknown sentiment: {sentiment}. Message discarded.")
#                 connection.close()
#                 return
#
#             # Publish the message with the routing key
#             channel.basic_publish(exchange='sentiment_exchange', routing_key=routing_key, body=message_json)
#             print(f'Successfully sent message: {message_json} with routing key: {routing_key}')
#
#             # Close connection
#             connection.close()
#
#         except Exception as e:
#             print(f"Error sending message to RabbitMQ: {e}")
#
#     def process_and_send(self, text):
#         # Predict sentiment
#         sentiment = self.predict_sentiment(text)
#
#         # Send the sentiment to RabbitMQ
#         self.send_message_to_rabbitmq(text, sentiment)

import json
from a_connection import get_rabbitmq_connection
import torch
import transformers
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

class SentimentAnalysisAgent:
    def __init__(self, distilbert_path='D:/Data/PyCharmProjects/MAS-sentiment-analysis/sent_analysis/models/DB_n_LR/DB.pth',
                 logistic_model_path='D:/Data/PyCharmProjects/MAS-sentiment-analysis/sent_analysis/models/DB_n_LR/logistic_regression_model.pkl'):
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
        tokenized = self.tokenizer.encode(text, add_special_tokens=True)
        max_len = 512
        padded_token_embeddings = np.array(tokenized + [0] * (max_len - len(tokenized)))
        attention_mask = np.where(padded_token_embeddings != 0, 1, 0)
        return padded_token_embeddings, attention_mask

    def get_bert_features(self, text):
        padded_token_embeddings, attention_mask = self.preprocess_text(text)
        input_ids = torch.tensor([padded_token_embeddings])
        attention_mask = torch.tensor([attention_mask])
        with torch.no_grad():
            outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        features = outputs[0][:, 0, :].numpy()
        return features

    def predict_sentiment(self, text):
        features = self.get_bert_features(text)
        sentiment = self.lr_model.predict(features)
        print(f"Predicted sentiment for text '{text}': {sentiment[0]}")
        return int(sentiment[0])

    def process_message(self, body):
        message = json.loads(body)
        text = message.get('text', None)
        if not text:
            print(f"Malformed message received: {message}")
            return

        sentiment = self.predict_sentiment(text)
        self.route_message(text, sentiment)

    def route_message(self, text, sentiment):
        try:
            connection = get_rabbitmq_connection()
            channel = connection.channel()

            # Declare the exchange and publish result
            channel.exchange_declare(exchange='sentiment_exchange', exchange_type='direct')
            routing_key = {0: 'negative', 1: 'neutral', 2: 'positive'}.get(sentiment, None)
            if routing_key:
                channel.basic_publish(exchange='sentiment_exchange',
                                      routing_key=routing_key,
                                      body=json.dumps({'text': text, 'sentiment': sentiment}))
                print(f"Sent '{text}' with sentiment '{sentiment}' to '{routing_key}' queue")
            connection.close()

        except Exception as e:
            print(f"Error routing message: {e}")

    def start_consuming(self):
        try:
            connection = get_rabbitmq_connection()
            channel = connection.channel()
            channel.queue_declare(queue='input_queue')
            channel.basic_consume(queue='input_queue', on_message_callback=self.callback, auto_ack=True)
            print("Waiting for messages from Coordinator...")
            channel.start_consuming()
        except Exception as e:
            print(f"Error consuming messages: {e}")

    def callback(self, ch, method, properties, body):
        self.process_message(body)


if __name__ == '__main__':
    agent = SentimentAnalysisAgent()
    agent.start_consuming()
