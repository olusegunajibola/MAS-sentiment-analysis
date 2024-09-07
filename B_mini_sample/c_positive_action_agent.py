import pika

class PositiveActionAgent:
    def __init__(self):
        self.connection, self.channel = self.connect_to_rabbitmq()
        self.channel.queue_declare(queue='action')

    def connect_to_rabbitmq(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()
        return connection, channel

    def respond(self, ch, method, properties, body):
        sentiment = float(body.decode('utf-8'))
        if sentiment > 0:
            print("PositiveActionAgent: That's great! I'm glad to hear that.")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_listening(self):
        self.channel.basic_consume(queue='action', on_message_callback=self.respond)
        print('PositiveActionAgent is waiting for messages...')
        self.channel.start_consuming()

if __name__ == "__main__":
    positive_agent = PositiveActionAgent()
    positive_agent.start_listening()