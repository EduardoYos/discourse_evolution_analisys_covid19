# import matplotlib.pyplot as plt
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

#PART 3
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords, wordnet
from sklearn.decomposition import LatentDirichletAllocation


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

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def main():
    start_time = time.time()

#---------------------------------


#----------------------------








    initial_data_path = "data/prefeituras.csv"
    city_hall_df = pd.read_csv(
        initial_data_path, encoding="utf-8",
         dtype={'Sponsor Id': str,'Sponsor Name': str,'Sponsor Category': str}
    )
    for col in city_hall_df.columns:
        print(col, city_hall_df[col].isnull().sum())
    rws = city_hall_df
    rws = city_hall_df.head(10)

    rws['Message'] = rws['Message'].str.replace('[^\w\s.]', '', flags=re.UNICODE)
    rws['Message'] = rws['Message'].fillna('')

    rws['tokenized'] = rws['Message'].apply(lambda x: word_tokenize(text=x, language="portuguese"))
    # rws['tokenized'] = rws['Message'].apply(word_tokenize)
    rws['lower'] = rws['tokenized'].apply(lambda x: [word.lower() for word in x])
    punc = strin.punctuation
    rws['no_punc'] = rws['lower'].apply(lambda x: [word for word in x if word not in punc])
    stop_words = set(stopwords.words('portuguese'))
    rws['stopwords_removed'] = rws['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])
    rws['pos_tags'] = rws['stopwords_removed'].apply(nltk.tag.pos_tag)
    rws['wordnet_pos'] = rws['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    wnl = WordNetLemmatizer()
    rws['lemmatized'] = rws['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

    #rws.to_csv('data/pre_processed_discourses_10.csv')

# ------------------ pre visualization data ------------------- #
    df = rws[['Page Name', 'Post Created Date', 'lemmatized']]
    #df = df.sort_values(by="Post Created Date")
    df['Post Created Date'] = pd.to_datetime(df['Post Created Date'])
    df['Post Created Date'] = df['Post Created Date'].dt.strftime('%Y-%m')





    #--
    #--
    #--
    #--
    #------- PART 3 -----------------------------------------------------------------------------------#
    df = rws[['wordnet_pos', 'lemmatized']]
    df['lemma_str'] = [' '.join(map(str, l)) for l in df['lemmatized']]
    tf_vectorizer = CountVectorizer(max_df=0.9, min_df=25, max_features=5000)

    tf = tf_vectorizer.fit_transform(df['lemma_str'].values.astype('U'))  # convert to unicode, essentially string
    tf_feature_names = tf_vectorizer.get_feature_names()

    doc_term_matrix = pd.DataFrame(tf.toarray(), columns=list(tf_feature_names))
    print(doc_term_matrix)
    lda_model = LatentDirichletAllocation(n_components=10, learning_method='online',
                                          max_iter=500, random_state=0).fit(tf)
    no_top_words = 10
    display_topics(lda_model, tf_feature_names, no_top_words)


    print("\n Execution time: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
