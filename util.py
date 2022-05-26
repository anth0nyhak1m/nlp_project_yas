import pandas as pd
import re
#import spacy

def cleaning_data(true_df_path, fake_df_path):
    """
    Helper function to pre-process data (lemmatization, tokenization, removal of punctation & stop words).

    Returns a list of tokens 
    """
    #load in data
    fake_df = pd.read_csv(fake_df_path, usecols= ["text"])
    true_df = pd.read_csv(true_df_path, usecols= ["text"])

    #add labels
    fake_df['target'] = 0
    true_df['target'] = 1

    #remove newspaper source (Reuters) from the "true" articles before merging
    true_df['text'] = true_df['text'].replace(r'\A.*\(Reuters\)', '', regex=True) 

    # Merge fake and true articles into one dataset 
    data = true_df.append(fake_df).sample(frac=1).reset_index().drop(columns=['index'])

    #remove all punctuation & single letters
    data['text'] = data['text'].replace(r'[^\w\s]', '', regex=True)
    data['text'] = data['text'].replace(r'\s\w\s', '', regex=True)

    return data

