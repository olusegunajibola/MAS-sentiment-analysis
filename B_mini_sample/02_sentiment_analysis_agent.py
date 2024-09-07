from textblob import TextBlob

from a_connection import connect_to_rabbitmq
class SentimentAnalysisAgent:
    def __init__(self):
        self.connection, self.channel = connect_to_rabbitmq()
        self.channel.queue_declare(queue='sentiment')

    def analyze_sentiment(self, ch, method, properties, body):
        text = body.decode('utf-8')
        sentiment = TextBlob(text).sentiment.polarity
        print(f"Analyzed sentiment: {sentiment}")
        self.channel.basic_publish(exchange='', routing_key='action', body=str(sentiment))

    def start_listening(self):
        self.channel.basic_consume(queue='sentiment', on_message_callback=self.analyze_sentiment, auto_ack=True)
        print('Sentiment Analysis Agent is waiting for messages...')
        self.channel.start_consuming()

if __name__ == "__main__":
    sentiment_agent = SentimentAnalysisAgent()
    sentiment_agent.start_listening()
