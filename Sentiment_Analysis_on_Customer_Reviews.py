import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import argparse

# Download required resources
nltk.download('vader_lexicon')

# Sentiment Analysis Function
def analyze_sentiment(file_path, output_path):
    df = pd.read_csv(file_path)
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis
    df['sentiment_score'] = df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    
    # Categorize sentiments
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
    
    # Save results
    df.to_csv(output_path, index=False)
    print("Sentiment analysis complete. Results saved to", output_path)

# Command-line interface for execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment Analysis Tool')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('output_file', type=str, help='Path to save the analyzed CSV file')
    args = parser.parse_args()
    analyze_sentiment(args.input_file, args.output_file)