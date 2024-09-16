from b_sentiment_analysis_agent import SentimentAnalysisAgent

def main():
    # Instantiate the sentiment analysis agent
    agent = SentimentAnalysisAgent()

    # Sample text data
    text = "The company's profits are expected to rise this quarter."

    # Process the text and send to RabbitMQ
    agent.process_and_send(text)

if __name__ == '__main__':
    main()
