from textblob import TextBlob


# Sentiment Analysis Agent
class SentimentAnalysisAgent:
    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        return sentiment


# Action Agents
class PositiveActionAgent:
    def respond(self):
        return "That's great! I'm glad to hear that."


class NegativeActionAgent:
    def respond(self):
        return "I'm sorry to hear that. How can I help?"


class NeutralActionAgent:
    def respond(self):
        return "Thank you for sharing."


# Coordinator Agent
class CoordinatorAgent:
    def __init__(self):
        self.sentiment_agent = SentimentAnalysisAgent()
        self.positive_agent = PositiveActionAgent()
        self.negative_agent = NegativeActionAgent()
        self.neutral_agent = NeutralActionAgent()

    def process(self, text):
        sentiment = self.sentiment_agent.analyze_sentiment(text)

        if sentiment > 0:
            return self.positive_agent.respond()
        elif sentiment < 0:
            return self.negative_agent.respond()
        else:
            return self.neutral_agent.respond()


# Testing the system
if __name__ == "__main__":
    coordinator = CoordinatorAgent()
    # text = "I'm feeling great today!"
    text = "the weather is awfully cold"
    response = coordinator.process(text)
    print(response)
