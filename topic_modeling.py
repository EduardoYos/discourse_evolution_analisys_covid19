# import matplotlib.pyplot as plt
# import seaborn as sns
import time
import pandas as pd
import numpy as np
import nltk
import re
import string as strin
import seaborn as sns
import matplotlib.pyplot as plt

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# import fasttext
# import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# PART 3
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

# funcao para adicionar apenas parte do dataset (5 primeiras colunas a cada 100)
def logic(index):
    divisor = 10000
    while (1):
        if (index - divisor > 0):
            index -= divisor
        else:
            if (divisor > 1000):
                divisor /= 10
            else:
                break
    # if 5 < index:
    if 10 < index < 990:
        return True
    else:
        return False


def main():
    start_time = time.time()

    initial_data_path = "data/prefeituras.csv"
    city_hall_df = pd.read_csv(
        initial_data_path, encoding="utf-8",
        dtype={'Sponsor Id': str, 'Sponsor Name': str, 'Sponsor Category': str},
        skiprows=lambda x: logic(x)
    )
    # for col in city_hall_df.columns:
    #     print(col, city_hall_df[col].isnull().sum())

    print(city_hall_df.shape[0])

    rws = city_hall_df
    # rws = city_hall_df.head(10)

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

    # rws.to_csv('data/pre_processed_discourses_10.csv')

    # ------------------ pre visualization data ------------------- #

    # ------- PART 3 -----------------------------------------------------------------------------------#
    df = rws[['wordnet_pos', 'Post Created Date', 'lemmatized']]
    df['lemma_str'] = [' '.join(map(str, l)) for l in df['lemmatized']]
    tf_vectorizer = CountVectorizer(max_df=0.9, min_df=25, max_features=5000)

    tf = tf_vectorizer.fit_transform(df['lemma_str'].values.astype('U'))  # convert to unicode, essentially string
    tf_feature_names = tf_vectorizer.get_feature_names()

    doc_term_matrix = pd.DataFrame(tf.toarray(), columns=list(tf_feature_names))
    # print(doc_term_matrix)
    lda_model = LatentDirichletAllocation(n_components=5, learning_method='online',
                                          max_iter=500, random_state=0).fit(tf)
    no_top_words = 10
    display_topics(lda_model, tf_feature_names, no_top_words)

    lda_output = lda_model.fit_transform(tf)

    # # column names
    # topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    # # index names
    # n_documents = len(df)
    # df['docnames'] = ["Doc" + str(i) for i in range(n_documents)]
    #
    # # Make the pandas dataframe
    # df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=df['docnames'])
    # # print(df_document_topic)
    # if len(df) == len(df_document_topic):
    #     df_document_topic.insert(loc=0, column='post_created_date', value=df['Post Created Date'].values)
    #     df_document_topic['post_created_date'] = pd.to_datetime(df_document_topic['post_created_date'])
    #     df_document_topic['post_created_date'] = df_document_topic['post_created_date'].dt.strftime('%Y-%m')
    #     df_document_topic = df_document_topic.groupby('post_created_date').mean().reset_index()
    #     #df_document_topic['topic_list'] = df_document_topic[['Topic0', 'Topic1']].values.tolist()
    #     df_document_topic['topic_list'] = np.round(
    #         df_document_topic[['Topic0', 'Topic1', 'Topic2', 'Topic3', 'Topic4']],
    #         3).values.tolist()

    # column names
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    # index names
    n_documents = len(df)
    df['docnames'] = ["Doc" + str(i) for i in range(n_documents)]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=df['docnames'])
    if len(df) == len(df_document_topic):
        df_document_topic.insert(loc=0, column='post_created_date', value=df['Post Created Date'].values)
        df_document_topic['post_created_date'] = pd.to_datetime(df_document_topic['post_created_date'])
        df_document_topic['post_created_date'] = df_document_topic['post_created_date'].dt.strftime('%Y-%m')
        df_document_topic = df_document_topic.groupby('post_created_date').mean().reset_index()

        df = df_document_topic
        print(df)

        data = []   # post_created_date, topic, prob
        # iterate over rows (vertically)
        for index_row, row in df.iterrows():
            # iterate over columns (horizontally)
            for column_name, value in row.items():
                if column_name == 'post_created_date':
                    date = value
                else:
                    topic_name = column_name
                    topic_prob = value
                    data.append([date, topic_name, topic_prob])

        df_result = pd.DataFrame(data, columns=['post_created_date', 'topic', 'probability'])

        # seaborn graph
        # ax = sns.displot(df_result, x="post_created_date", hue="topic", y="probability")
        # ax = sns.plot(df_result, kind='bar', stacked=True, x="post_created_date", hue="topic", y="probability")
        df.set_index('post_created_date').plot(kind='bar', stacked=True, color=['blue', 'green', 'grey', 'red', 'orange'])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=5, fancybox=True, shadow=True)
        plt.subplots_adjust(bottom=0.20)

        plt.xticks(rotation=45)
        plt.show()

        #df_document_topic['topic_list'] = df_document_topic[['Topic0', 'Topic1']].values.tolist()
        # df_document_topic['topic_list'] = np.round(
        #     df_document_topic[['Topic0', 'Topic1', 'Topic2', 'Topic3', 'Topic4']],
        #     3).values.tolist()

    print("\n Execution time: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
