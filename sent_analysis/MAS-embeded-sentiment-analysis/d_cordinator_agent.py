# from b_sentiment_analysis_agent import SentimentAnalysisAgent
#
# def main():
#     # Instantiate the sentiment analysis agent
#     agent = SentimentAnalysisAgent()
#
#     # Sample text data
#     # text = "The company's profits are expected to rise this quarter."
#     # text = "Earnings are down by half this year. The losses are very much."
#     text = "Market experts are unmoved by the company's performance this quarter"
#     # print('Enter a sentence: ')
#     # text = input()
#     # Process the text and send to RabbitMQ
#     agent.process_and_send(text)
#
# if __name__ == '__main__':
#     main()

from a_connection import get_rabbitmq_connection
import json

def main():
    try:
        # Connect to RabbitMQ
        connection = get_rabbitmq_connection()
        channel = connection.channel()

        # Declare the input queue
        channel.queue_declare(queue='input_queue')

        # User input
        text = input("Enter a sentence: ")

        # Prepare and send the message
        message = {'text': text}
        channel.basic_publish(exchange='', routing_key='input_queue', body=json.dumps(message))
        print(f"Sent text to Sentiment Analysis Agent: {text}")

        # Close connection
        connection.close()

    except Exception as e:
        print(f"Error sending message: {e}")

if __name__ == '__main__':
    main()

