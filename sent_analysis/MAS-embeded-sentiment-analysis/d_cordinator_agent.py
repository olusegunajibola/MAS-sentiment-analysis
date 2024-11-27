from a_connection import get_rabbitmq_connection
import json
import time

def main():
    try:
        # Connect to RabbitMQ
        connection = get_rabbitmq_connection()
        channel = connection.channel()

        # Declare the input queue
        channel.queue_declare(queue='input_queue')

        # User input
        text = input("Enter a sentence: ")
        start_time = time.time()

        # Prepare and send the message
        message = {'text': text, 'timestamp': start_time}
        channel.basic_publish(exchange='', routing_key='input_queue', body=json.dumps(message))
        print(f"Sent text to Sentiment Analysis Agent at {time.ctime(start_time)}: {text}")

        # Close connection
        connection.close()

    except Exception as e:
        print(f"Error sending message: {e}")

if __name__ == '__main__':
    main()


# from a_connection import get_rabbitmq_connection
# import json
#
# def main():
#     try:
#         # Connect to RabbitMQ
#         connection = get_rabbitmq_connection()
#         channel = connection.channel()
#
#         # Declare the input queue
#         channel.queue_declare(queue='input_queue')
#
#         # User input
#         text = input("Enter a sentence: ")
#
#         # Prepare and send the message
#         message = {'text': text}
#         channel.basic_publish(exchange='', routing_key='input_queue', body=json.dumps(message))
#         print(f"Sent text to Sentiment Analysis Agent: {text}")
#
#         # Close connection
#         connection.close()
#
#     except Exception as e:
#         print(f"Error sending message: {e}")
#
# if __name__ == '__main__':
#     main()
#