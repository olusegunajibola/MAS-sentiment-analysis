# import json
# from a_connection import get_rabbitmq_connection
# 
# def callback(ch, method, properties, body):
#     try:
#         message = json.loads(body)
#         sentiment = message.get('sentiment', None)
#         text = message.get('text', 'No text provided')
# 
#         print(f"Neutral action for text: '{text}'")
#     except Exception as e:
#         print(f"Error processing message: {e}")
# 
# def main():
#     try:
#         # Setup RabbitMQ connection
#         connection = get_rabbitmq_connection()
#         channel = connection.channel()
# 
#         # Declare a direct exchange
#         channel.exchange_declare(exchange='sentiment_exchange', exchange_type='direct')
# 
#         # Declare and bind a queue for neutral sentiment
#         queue_name = 'neutral_queue'
#         channel.queue_declare(queue=queue_name)
#         channel.queue_bind(exchange='sentiment_exchange', queue=queue_name, routing_key='neutral')
# 
#         # Start consuming messages
#         channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
#         print('Waiting for neutral sentiment messages...')
#         channel.start_consuming()
# 
#     except Exception as e:
#         print(f"Error with RabbitMQ connection or consumption: {e}")
# 
# if __name__ == '__main__':
#     main()

import json
from a_connection import get_rabbitmq_connection
import time

def callback(ch, method, properties, body):
    start_time = time.time()
    message = json.loads(body)
    text = message.get('text', '')
    sentiment = message.get('sentiment', None)

    print(f"Neutral Action Agent received message at {time.ctime(start_time)}: {message}")

    # Simulate processing
    # time.sleep(1)  # Optional: Simulate some processing delay

    end_time = time.time()
    print(f"Finished processing neutral sentiment at {time.ctime(end_time)}. Processing time: {end_time - start_time:.2f} seconds.")

def main():
    connection = get_rabbitmq_connection()
    channel = connection.channel()
    channel.exchange_declare(exchange='sentiment_exchange', exchange_type='direct')
    channel.queue_declare(queue='neutral_queue')
    channel.queue_bind(exchange='sentiment_exchange', queue='neutral_queue', routing_key='neutral')
    channel.basic_consume(queue='neutral_queue', on_message_callback=callback, auto_ack=True)
    print("Waiting for neutral sentiment messages...")
    channel.start_consuming()

if __name__ == '__main__':
    main()
