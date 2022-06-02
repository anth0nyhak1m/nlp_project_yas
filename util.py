import pandas as pd

#imports to remove stop words 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

#import to do word stemming
from nltk.stem.snowball import EnglishStemmer

#imports to do wordcloud visualization of features 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#for tf/idf
from sklearn.feature_extraction.text import TfidfVectorizer

def cleaning_data(true_df_path, fake_df_path, attribute_to_use, bert = True):
    """
    Helper function to pre-process data (lower case, remove article source, removal of punctation & stop words).

    Inputs:
      true_df_path: (str) path to csv file containing articles labeled as 'True'
      fake_df_path: (str) path to csv file containing articles labeled as 'False'

    Returns a pd dataframe with clean strings
    """
    #load in data
    fake_df = pd.read_csv(fake_df_path, usecols= [attribute_to_use])
    true_df = pd.read_csv(true_df_path, usecols= [attribute_to_use])

    #add labels
    fake_df['target'] = 0
    true_df['target'] = 1

    #remove newspaper source (Reuters) from the "true" articles before merging
    if attribute_to_use == 'text':
        true_df[attribute_to_use] = true_df[attribute_to_use].replace(r'\A.*\(Reuters\)', '', regex=True) 
    # i.e. if we are using titles
    else:
        true_df[attribute_to_use] = true_df[attribute_to_use].replace(r'Reuters', ' ', regex=True)

    # Merge fake and true articles into one dataset 
    data = true_df.append(fake_df).sample(frac=1).reset_index().drop(columns=['index'])

    if not bert:
        #remove all punctuation, single letters, and digits
        data[attribute_to_use] = data[attribute_to_use].replace(r'[^\w\s]', ' ', regex=True).str.lower()
        data[attribute_to_use] = data[attribute_to_use].replace(r'\s\w\s', ' ', regex=True)
        data[attribute_to_use] = data[attribute_to_use].replace(r'\d+\w*', ' ', regex=True)

        #remove stop words
        data[attribute_to_use] = data[attribute_to_use].apply(lambda words: ' '.join\
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


def wordcloud(series_of_strings, max_words): 
    """
    Visulalizes pandas series as a cloud of words.

    Returns: plot
    """
    word_cloud = WordCloud(collocations = False, background_color = 'white',\
                max_words = max_words, max_font_size = 100, width = 800,\
                height = 400).generate(str(series_of_strings))
    
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def tf_idf_vectorizer(train_attribute, val_attribute, test_attribute, max_features): 
    """vectorize training, validation, and testing data. Produce TF-IDF vectors for each subset"""

    # Note, ngrams = 1, which is the default value if not specified in TfidfVectorizer. 
    text_transformer = TfidfVectorizer(stop_words='english', max_features = max_features)

    # fit_transform() method learns vocabulary and `IDF` used for both training & test data. 
    # Returns document-term matrix with calculated `TF-IDF` values.
    X_train_text = text_transformer.fit_transform(train_attribute)

    # transform() method uses the vocabulary and document frequencies (df) learned by fit_transform(). 
    # Returns document-term matrix with calculated `TF-IDF` values.
    X_val_text = text_transformer.transform(val_attribute)
    X_test_text = text_transformer.transform(test_attribute)

    return text_transformer, X_train_text, X_val_text, X_test_text

    