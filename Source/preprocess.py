import pandas as pd
import os
from sklearn.model_selection import train_test_split
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_dataset(PATH='Dataset/stock_news/stock_news.csv'):
    
    train_path = 'Source/Data/Train/train_stock_news.csv'
    test_path = 'Source/Data/Test/test_stock_news.csv'
    valid_path = 'Source/Data/Valid/valid_stock_news.csv'
    
    
    df = pd.read_csv(os.path.join(os.getcwd(), PATH))
    
    # Count the number of samples for each sentiment category
    sentiment_counts = df['label'].value_counts()

    # Convert the Series object into a DataFrame
    sentiment_counts_df = sentiment_counts.reset_index()
    sentiment_counts_df.columns = ['sentiment', 'count']

    # Display the summary DataFrame
    print(sentiment_counts_df)
    
    for index, row in df.iterrows():
        
        if row['label'] == 'Negative':
            df.at[index, 'label'] = 0
        
        if row['label'] == 'Positive':
            df.at[index, 'label'] = 2
        
        if row['label'] == 'Neutral':
            df.at[index, 'label'] = 1
            
    df['label'] = df['label'].astype(int)
    
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)
    
    df_train.to_csv(os.path.join(os.getcwd(), train_path))
    
    df_test.to_csv(os.path.join(os.getcwd(), test_path))
    
    df_val.to_csv(os.path.join(os.getcwd(), valid_path))
    sentiment_counts = df_train['label'].value_counts()

    # Convert the Series object into a DataFrame
    sentiment_counts_df = sentiment_counts.reset_index()
    sentiment_counts_df.columns = ['sentiment', 'count']

    # Display the summary DataFrame
    print(sentiment_counts_df)
    
    return df

def clean_tokens(df, vocab):
    df['length'] = df.tokens.apply(len)
    df['clean_tokens'] = df.tokens.apply(lambda x: [t for t in x if t in vocab.freqs.keys()])
    df['clean_length'] = df.clean_tokens.apply(len)
    return df

def tokenize_text(df):
    df["tokens"] = df.headline.str.lower().str.strip().apply(lambda x: [token.text.strip() for token in nlp(x) if token.text.isalnum()])
    return df

def save(df, path):
    # train_path = os.path.join('Source', 'Data', 'Train', 'train_stock_news_tokens.csv')
    # # test_path = 'Source/Data/Test/test_stock_news.csv'
    # test_path = os.path.join('Source', 'Data', 'Test', 'test_stock_news_tokens.csv')

    # # valid_path = 'Source/Data/Valid/valid_stock_news.csv'
    # valid_path = os.path.join('Source', 'Data', 'Valid', 'valid_stock_news_tokens.csv')
    
    df.to_csv(os.path.join(os.getcwd(), path))
    
    # df_test.to_csv(os.path.join(os.getcwd(), test_path))
    
    # df_val.to_csv(os.path.join(os.getcwd(), valid_path))
    
    
def load(path):
    
    # train_path = 'Source/Data/Train/train_stock_news.csv'
    # train_path = os.path.join('Source', 'Data', 'Train', 'train_stock_news.csv')
    # # test_path = 'Source/Data/Test/test_stock_news.csv'
    # test_path = os.path.join('Source', 'Data', 'Test', 'test_stock_news.csv')

    # # valid_path = 'Source/Data/Valid/valid_stock_news.csv'
    # valid_path = os.path.join('Source', 'Data', 'Valid', 'valid_stock_news.csv')

    
    df= pd.read_csv(os.path.join(os.getcwd(), path))
    
    # df_test = pd.read_csv(os.path.join(os.getcwd(), test_path))
    
    # df_val = pd.read_csv(os.path.join(os.getcwd(), valid_path))
    
    return df
    
if __name__ == '__main__':
    df = preprocess_dataset()
    


