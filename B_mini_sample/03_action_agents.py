from a_connection import connect_to_rabbitmq

class PositiveActionAgent:
    def __init__(self):
        self.connection, self.channel = connect_to_rabbitmq()
        self.channel.queue_declare(queue='action')

    def respond(self, ch, method, properties, body):
        sentiment = float(body.decode('utf-8'))
        if sentiment > 0:
            print("PositiveActionAgent: That's great! I'm glad to hear that.")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_listening(self):
        self.channel.basic_consume(queue='action', on_message_callback=self.respond)
        print('PositiveActionAgent is waiting for messages...')
        self.channel.start_consuming()

class NegativeActionAgent:
    def __init__(self):
        self.connection, self.channel = connect_to_rabbitmq()
        self.channel.queue_declare(queue='action')

    def respond(self, ch, method, properties, body):
        sentiment = float(body.decode('utf-8'))
        if sentiment < 0:
            print("NegativeActionAgent: I'm sorry to hear that. How can I help?")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_listening(self):
        self.channel.basic_consume(queue='action', on_message_callback=self.respond)
        print('NegativeActionAgent is waiting for messages...')
        self.channel.start_consuming()

# Similar agent can be created for neutral sentiment
