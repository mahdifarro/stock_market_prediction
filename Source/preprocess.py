import pandas as pd
import os
from sklearn.model_selection import train_test_split

def preprocess_dataset(PATH='Dataset/stock_news/stock_news.csv'):
    
    train_path = 'Source/Data/Train/train_stock_news.csv'
    test_path = 'Source/Data/Test/test_stock_news.csv'
    valid_path = 'Source/Data/Valid/valid_stock_news.csv'
    
    
    df = pd.read_csv(os.path.join(os.getcwd(), PATH))
    

    
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
    
    return df
  
def load():
    
    train_path = 'Source/Data/Train/train_stock_news.csv'
    test_path = 'Source/Data/Test/test_stock_news.csv'
    valid_path = 'Source/Data/Valid/valid_stock_news.csv'
    
    df_train = pd.read_csv(os.path.join(os.getcwd(), train_path))
    
    df_test = pd.read_csv(os.path.join(os.getcwd(), test_path))
    
    df_val = pd.read_csv(os.path.join(os.getcwd(), valid_path))
    
    return df_train, df_test, df_val
    
if __name__ == '__main__':
    df = preprocess_dataset()
    print(df)


