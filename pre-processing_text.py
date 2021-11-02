# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
import time
import pandas as pd
import numpy as np
import nltk
import re
import string as strin
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# import fasttext
# import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
# plt.xticks(rotation=70)
# pd.options.mode.chained_assignment = None
# pd.set_option('display.max_colwidth', 100)
# %matplotlib inline

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def main():
    start_time = time.time()

    initial_data_path = "data/prefeituras.csv"
    city_hall_df = pd.read_csv(
        initial_data_path, encoding="utf-8",
         dtype={'Sponsor Id': str,'Sponsor Name': str,'Sponsor Category': str}
    )
    for col in city_hall_df.columns:
        print(col, city_hall_df[col].isnull().sum())
    # rws = city_hall_df.head(1000)
    rws = city_hall_df

    rws['Message'] = rws['Message'].str.replace('[^\w\s.]', '', flags=re.UNICODE)
    rws['Message'] = rws['Message'].fillna('')

    rws['tokenized'] = rws['Message'].apply(lambda x: word_tokenize(text=x, language="portuguese"))
    rws['lower'] = rws['tokenized'].apply(lambda x: [word.lower() for word in x])
    punc = strin.punctuation
    rws['no_punc'] = rws['lower'].apply(lambda x: [word for word in x if word not in punc])
    stop_words = set(stopwords.words('portuguese'))
    rws['stopwords_removed'] = rws['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])
    rws['pos_tags'] = rws['stopwords_removed'].apply(nltk.tag.pos_tag)
    rws['wordnet_pos'] = rws['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    wnl = WordNetLemmatizer()
    rws['lemmatized'] = rws['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    rws.to_csv('data/pre_processed_discourses.csv')

    print("Execution time: %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
