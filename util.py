import pandas as pd

#imports to remove stop words 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

#import to do word stemming
from nltk.stem.snowball import EnglishStemmer


def cleaning_data(true_df_path, fake_df_path):
    """
    Helper function to pre-process data (lower case, remove article source, removal of punctation & stop words).

    Inputs:
      true_df_path: (str) path to csv file containing articles labeled as 'True'
      fake_df_path: (str) path to csv file containing articles labeled as 'False'

    Returns a pd dataframe with clean strings
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
    data['text'] = data['text'].replace(r'[^\w\s]', ' ', regex=True).str.lower()
    data['text'] = data['text'].replace(r'\s\w\s', ' ', regex=True)

    #remove stop words
    data['text'] = data["text"].apply(lambda words: ' '.join\
                    (word for word in words.split() if word not in stop))

    return data


def stemming(pd_series):
    """
    Converts words into stems and returns a string with stemmed words per observation.
      
    Input:
      pd_series: (pd series) a pandas series where each observation represents an article in 
        a string form
    
    Returns: (pd series)
    """ 
    stemmer = EnglishStemmer()
    processed_series = pd_series.apply(lambda words: ' '.join(stemmer.stem(word) for word in words.split()))

    return processed_series