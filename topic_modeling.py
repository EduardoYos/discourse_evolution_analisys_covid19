# import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import nltk
import re
import string as strin
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# download when necessary:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pyLDAvis
import pyLDAvis.sklearn

TRANSFORM = True
GENERATE_GRAPH = True

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords, wordnet
from sklearn.decomposition import LatentDirichletAllocation


def set_df_date(df):
    df['day'] = pd.to_datetime(df['Post Created Date'])
    df['ano_mes'] = df['day'].dt.to_period('M')
    df['week'] = df['day'].dt.week
    df['month'] = df['day'].dt.month
    df['year'] = df['day'].dt.year
    return df


# def mayor_to_city(x, df):
#     return df[df['mayor'] == x].city.values[0]
#
#
# def page_name_to_city(x):
#     return x.replace('do ', 'de ').partition(' de ')[2].lower()


def get_covid_posts(df, covid_words):
    len_total = len(df)
    df = df[df['Message'].str.contains("|".join(covid_words), case=False, na=False)]
    len_covid = len_total - len(df)
    print('{} posts foram removidos, {:.2f}% do total'.format(len_covid, (len_covid/len_total)*100))
    return df


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
    warnings.filterwarnings("ignore")

    # region New DFs
    path = './data/'
    df_city_halls = pd.read_csv(path + 'prefeituras.csv')
    df_mayor_city = pd.read_csv(path + 'prefeito-cidade.csv')
    df_20 = pd.read_csv(path + 'prefeitos2020.csv')
    df_21 = pd.read_csv(path + 'prefeitos2021.csv')

    # df_city_halls['city'] = df_city_halls['Page Name'].apply(page_name_to_city)
    # df_city_halls['city'] = df_city_halls['Page Name'].apply(page_name_to_city)
    df_city_halls['city'] = df_city_halls['Page Name'].apply(
        lambda x: x.replace('do ', 'de ').partition(' de ')[2].lower())

    # df_20['city'] = df_20['Page Name'].apply(lambda x: mayor_to_city(x, df_mayor_city))
    df_20['city'] = df_20['Page Name'].apply(lambda x: df_mayor_city[df_mayor_city['mayor'] == x].city.values[0])
    # df_21['city'] = df_21['Page Name'].apply(lambda x: mayor_to_city(x, df_mayor_city))
    df_21['city'] = df_21['Page Name'].apply(lambda x: df_mayor_city[df_mayor_city['mayor'] == x].city.values[0])

    df_mayors = pd.concat([df_20, df_21])
    df_last = pd.concat([df_city_halls[['Message', 'city', 'Post Created Date']],
                          df_mayors[['Message', 'city', 'Post Created Date']]])

    df_last = set_df_date(df_last)
    # endregion

    list_covid_words = ['comorbidade', 'ivermectina', 'tamiflu', 'azitromicina', 'lockdown', 'distanciamento social',
                   'hidroxicloroquina', 'covid19', 'isolamento', 'cloroquina', 'quarentena', 'covid_19', 'coronavirus',
                   'pandemia', 'corona', 'covid']
    df_last = get_covid_posts(df_last, list_covid_words)
    # print(df_last['Message'][54])
    # print(df_last['Message'][0])
    # removing http/https links
    df_last['Message'] = df_last['Message'].str.replace(r'(?:http|https)\:\/\/[a-zA-Z0-9\.\/\?\:@\-_=#]+', '')
    # char_list = ['#']
    # res = [word for word in df_last['Message'] if all(ch not in word for ch in char_list)]

    # print(df_last['Message'][54])
    # print(df_last['Message'][0])

    print('Total de posts: {}'.format(df_last.shape[0]))

    rws = df_last
    # initial_data_path = "data/prefeituras.csv"
    # city_hall_df = pd.read_csv(
    #     initial_data_path, encoding="utf-8",
    #     dtype={'Sponsor Id': str, 'Sponsor Name': str, 'Sponsor Category': str}
    #     # skiprows=lambda x: logic(x)
    # )
    # # for col in city_hall_df.columns:
    # #     print(col, city_hall_df[col].isnull().sum())
    #
    # print(city_hall_df.shape[0])
    #
    # rws = city_hall_df
    # # rws = city_hall_df.head(100)

    rws['Message'] = rws['Message'].str.replace('[^\w\s.]', '', flags=re.UNICODE)
    rws['Message'] = rws['Message'].fillna('')

    # rws['tokenized'] = rws['Message'].apply(lambda x: word_tokenize(text=x, language="portuguese"))
    # rws['lower'] = rws['tokenized'].apply(lambda x: [word.lower() for word in x])
    # punc = strin.punctuation
    # rws['no_punc'] = rws['lower'].apply(lambda x: [word for word in x if word not in punc])
    # stop_words = set(stopwords.words('portuguese'))  # remover outras palavras
    # rws['stopwords_removed'] = rws['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])
    # rws['pos_tags'] = rws['stopwords_removed'].apply(nltk.tag.pos_tag)
    # rws['wordnet_pos'] = rws['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    # wnl = WordNetLemmatizer()  # diferente
    # rws['lemmatized'] = rws['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    rws['tokenized'] = rws['Message'].apply(lambda x: word_tokenize(text=x, language="portuguese"))
    # rws['lower'] = rws['tokenized'].apply(lambda x: [word.lower() for word in x])
    # punc = strin.punctuation
    # rws['no_punc'] = rws['tokenized'].apply(lambda x: [word for word in x if word not in punc])
    # stop_words = set(stopwords.words('portuguese'))  # remover outras palavras
    # rws['stopwords_removed'] = rws['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])
    rws['pos_tags'] = rws['tokenized'].apply(nltk.tag.pos_tag)
    rws['wordnet_pos'] = rws['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    wnl = WordNetLemmatizer()  # diferente
    rws['lemmatized'] = rws['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

    # rws.to_csv('data/pre_processed_discourses_10.csv', sep=';')

    # vectorizer
    stop_words = stopwords.words('portuguese')  # removing stop words
    stop_words.extend((list(rws.city.unique())))
    stop_words.extend(['pracegover', 'macapa', 'acre', 'natal', 'belém',
                       'curitiba', 'imagem', 'texto', 'fortalezar', 'foto'])

    print("\nExecution time - first part: %s seconds" % (time.time() - start_time))
    print("\nInitiating second part...")
    # ------- PART 2 -----------------------------------------------------------------------------------#
    df = rws[['Post Created Date', 'lemmatized']]
    df['lemma_str'] = [' '.join(map(str, l)) for l in df['lemmatized']]
    # count_vectorizer = CountVectorizer(max_df=0.9, min_df=25, max_features=5000)  # diferente
    count_vectorizer = CountVectorizer(# max_df=0.9,
                                       min_df=10,
                                       analyzer='word',
                                       stop_words=stop_words,  # remove stop words
                                       lowercase=True,  # convert all words to lowercase
                                       token_pattern=r'[(?u)\b\w\w+\b]{3,}',  # select only words where nº chars > 3
                                    )

    transf_matrix = count_vectorizer.fit_transform(df['lemma_str'].values.astype('U'))  # convert to unicode, essentially string
    lda_model = LatentDirichletAllocation(n_components=5, learning_method='online',
                                          max_iter=10, random_state=0,
                                          n_jobs=-1,  # Use all available CPUs
                                          ).fit(transf_matrix)

    tf_feature_names = count_vectorizer.get_feature_names()
    # doc_term_matrix = pd.DataFrame(transf_matrix.toarray(), columns=list(tf_feature_names))
    # print(doc_term_matrix)
    no_top_words = 30
    print("Printing topics... \n")
    display_topics(lda_model, tf_feature_names, no_top_words)


    panel = pyLDAvis.sklearn.prepare(lda_model, transf_matrix, count_vectorizer, mds='tsne')
    #panel

    pyLDAvis.save_html(panel, './data/lda_result_mine.html')

    # Transform dataframe
    if (TRANSFORM):
        lda_output = lda_model.fit_transform(transf_matrix)

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
            # print(df)

            data = []  # post_created_date, topic, prob
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
        if (GENERATE_GRAPH):
            df.set_index('post_created_date').plot(kind='bar', stacked=True,
                                                   color=['blue', 'green', 'grey', 'red', 'orange'])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                       ncol=5, fancybox=True, shadow=True)
            plt.subplots_adjust(bottom=0.20)
            plt.xlabel("Mês de criação")
            plt.ylabel("Probabilidade de cada Tópico")

            plt.xticks(rotation=45)
            print("saving graph...")
            plt.savefig('./data/plot_graph.png')
            print("graph Saved!")
            plt.show()



        # df_document_topic['topic_list'] = df_document_topic[['Topic0', 'Topic1']].values.tolist()
        # df_document_topic['topic_list'] = np.round(
        #     df_document_topic[['Topic0', 'Topic1', 'Topic2', 'Topic3', 'Topic4']],
        #     3).values.tolist()

    print("\nExecution time: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
