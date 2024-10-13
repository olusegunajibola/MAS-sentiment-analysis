import os
import pandas as pd
from groq import Groq


def augment_sentiment(df, target_ratio, sentiment_class, batch_size=10, sentiment_column='sentiment', text_column='news'):
    """
    Augment an underrepresented sentiment class in a DataFrame using Groq API.

    Args:
        df (pd.DataFrame): Original DataFrame with sentiment and text data.
        target_ratio (float): Multiplier for augmenting the underrepresented class.
        sentiment_class (str): The sentiment class to augment (e.g., 'negative').
        sentiment_column (str): Column containing the sentiment labels (default 'sentiment').
        text_column (str): Column containing the text/news data (default 'news').

    Returns:
        pd.DataFrame: A new DataFrame with augmented data added to balance the sentiment class.
    """
    # # batch size based on API limitations
    # batch_size = batch_size

    # Set up your Groq client (ensure your API key is set in the environment)
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    # Find the underrepresented texts (those with the specified sentiment_class)
    underrepresented_texts = df[df[sentiment_column] == sentiment_class][text_column].tolist()

    # Number of rows we want for the sentiment class
    current_rows = len(underrepresented_texts)
    target_rows = int(current_rows * target_ratio)

    # Number of additional examples needed
    needed_examples = target_rows - current_rows

    # Augment the underrepresented class with new examples
    augmented_texts = []
    for i in range(needed_examples):
        # Select a random text from the underrepresented class to augment
        text = underrepresented_texts[i % current_rows]

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data augmentation assistant."},
                {"role": "user",
                 "content": f"Generate a headline similar to: {text} and reply with response only without quotes"},
            ],
            model="llama3-8b-8192"
        )

        # Get the augmented text from the response
        augmented_data = response.choices[0].message.content
        augmented_texts.append(augmented_data)

    # Create a new DataFrame for the augmented data
    augmented_df = pd.DataFrame({
        text_column: augmented_texts,
        sentiment_column: [sentiment_class] * needed_examples  # Label the new examples with the same sentiment class
    })

    # Combine the original DataFrame with the augmented data
    balanced_df = pd.concat([df, augmented_df], ignore_index=True)

    return balanced_df, augmented_df


# # Example usage
# df = pd.read_csv("D:/Data/PyCharmProjects/MAS-sentiment-analysis/data/financial_news.csv",
#                  names=['sentiment', 'news'])
#
# # Augment 'negative' sentiment class to have 4x the current number of rows
# balanced_df = augment_sentiment(df, target_ratio=4, sentiment_class='negative')
#
# # Display the balanced DataFrame
# print(balanced_df)
