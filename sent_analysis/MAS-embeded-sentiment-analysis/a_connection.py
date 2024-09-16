import pika

def get_rabbitmq_connection():
    return pika.BlockingConnection(pika.ConnectionParameters('localhost'))
