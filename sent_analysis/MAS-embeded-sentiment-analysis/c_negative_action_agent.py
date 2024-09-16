import pika
import json

def callback(ch, method, properties, body):
    message = json.loads(body)
    sentiment = message['sentiment']
    # if sentiment == 'negative':
    if sentiment == 0:
        print(f"Negative action for text: {message['text']}")

# Setup RabbitMQ connection
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare the sentiment queue
channel.queue_declare(queue='sentiment')

# Start consuming messages
channel.basic_consume(queue='sentiment', on_message_callback=callback, auto_ack=True)
print('Waiting for negative sentiment messages...')
channel.start_consuming()
