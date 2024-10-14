import matplotlib.pyplot as plt
def pie_plot(data, title):
    # Calculate counts and percentages
    value_counts = data['sentiment'].value_counts()
    percentages = value_counts / value_counts.sum() * 100

    # Create labels with both values and percentages
    labels = [f'{label}: {count}' for label, count in zip(value_counts.index, value_counts)]

    # Define the explode parameter to separate the "negative" slice
    explode = [0.05 if label == 'negative' else 0 for label in value_counts.index]

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(value_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors,
            explode=explode)
    plt.title(title)
    plt.show()