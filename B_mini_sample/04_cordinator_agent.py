from a_connection import connect_to_rabbitmq

class CoordinatorAgent:
    def __init__(self):
        self.connection, self.channel = connect_to_rabbitmq()
        self.channel.queue_declare(queue='sentiment')

    def send_text_for_analysis(self, text):
        self.channel.basic_publish(exchange='', routing_key='sentiment', body=text)
        print(f"Sent text for analysis: {text}")

if __name__ == "__main__":
    coordinator = CoordinatorAgent()
    coordinator.send_text_for_analysis("I am so happy with the service!")
    coordinator.send_text_for_analysis("This is terrible, I'm very upset.")
