# # import pika
# import json
# from a_connection import get_rabbitmq_connection
#
# def callback(ch, method, properties, body):
#     message = json.loads(body)
#     sentiment = message['sentiment']
#     # if sentiment == 'positive':
#     if sentiment == 2:
#         print(f"Positive action for text: {message['text']}")
#
# # Setup RabbitMQ connection
# # connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
# connection = get_rabbitmq_connection()
# channel = connection.channel()
#
# # Declare the sentiment queue
# channel.queue_declare(queue='sentiment')
#
# # Start consuming messages
# channel.basic_consume(queue='sentiment', on_message_callback=callback, auto_ack=True)
# print('Waiting for positive sentiment messages...')
# channel.start_consuming()

import json
from a_connection import get_rabbitmq_connection

def callback(ch, method, properties, body):
    try:
        message = json.loads(body)
        sentiment = message.get('sentiment', None)
        text = message.get('text', 'No text provided')

        print(f"positive action for text: '{text}'")
    except Exception as e:
        print(f"Error processing message: {e}")

def main():
    try:
        # Setup RabbitMQ connection
        connection = get_rabbitmq_connection()
        channel = connection.channel()

        # Declare a direct exchange
        channel.exchange_declare(exchange='sentiment_exchange', exchange_type='direct')

        # Declare and bind a queue for positive sentiment
        queue_name = 'positive_queue'
        channel.queue_declare(queue=queue_name)
        channel.queue_bind(exchange='sentiment_exchange', queue=queue_name, routing_key='positive')

        # Start consuming messages
        channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
        print('Waiting for positive sentiment messages...')
        channel.start_consuming()

    except Exception as e:
        print(f"Error with RabbitMQ connection or consumption: {e}")

if __name__ == '__main__':
    main()


