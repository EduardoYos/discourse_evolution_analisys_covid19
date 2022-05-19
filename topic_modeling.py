# main file of LDA topics modeling. It gets the city halls dataset and treated them to do the LDA
# topics modeling. It cleans all the data, filtering the covid posts and doing all the steps of
# LDA topics modeling. Then, the script creates a file of documents-topics relation and also generate a graph
# of topics during the months of the pandemic.

# input file: "prefeituras.csv" -> main dataset with city halls posts and all other information about the posts
# input file: "prefeito-cidade.csv" -> dataset with relation of mayors-city halls of Brazil
# input file: "prefeitos2020.csv" -> dataset with posts by mayors of Brazilians cities in 2020
# input file: "prefeitos2021.csv" -> dataset with posts by mayors of Brazilians cities in 2021

# output file: "lda_result_mine.html" -> result of pyLDAvis visualization of topics
# output file: "document_topic.csv" -> dataset with probabilities of each topic order by month
# output file: "plot_graph.png" -> basic plot of topics during the months of the pandemic.

import time
import pandas as pd
import numpy as np
import nltk
import re
import string as strin
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from datetime import datetime


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
GENERATE_GRAPH = False
SAVE = True

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
    df_last = df_last[(df_last['Post Created Date'] >= '2020-04-01') & (df_last['Post Created Date'] < '2021-01-01')].reset_index(drop=True)

    # endregion

    list_covid_words = ['comorbidade', 'ivermectina', 'tamiflu', 'azitromicina', 'lockdown', 'distanciamento social',
                   'hidroxicloroquina', 'covid19', 'isolamento', 'cloroquina', 'quarentena', 'covid_19', 'coronavirus',
                   'pandemia', 'corona', 'covid']
    df_last = get_covid_posts(df_last, list_covid_words)
    # removing http/https links
    df_last['Message'] = df_last['Message'].str.replace(r'(?:http|https)\:\/\/[a-zA-Z0-9\.\/\?\:@\-_=#]+', '')

    print('Total de posts: {}'.format(df_last.shape[0]))

    rws = df_last
    # rws = df_last.head(10)

    rws['Message'] = rws['Message'].str.replace('[^\w\s.]', '', flags=re.UNICODE)
    rws['Message'] = rws['Message'].fillna('')
    rws['tokenized'] = rws['Message'].apply(lambda x: word_tokenize(text=x, language="portuguese"))
    rws['pos_tags'] = rws['tokenized'].apply(nltk.tag.pos_tag)
    rws['wordnet_pos'] = rws['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    wnl = WordNetLemmatizer()  # diferente
    rws['lemmatized'] = rws['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

    #if SAVE:
        # rws.to_csv('data/pre_processed_discourses_10.csv', sep=';')

    # vectorizer
    stop_words = stopwords.words('portuguese')  # removing stop words
    stop_words.extend((list(rws.city.unique())))
    stop_words.extend(['pracegover', 'macapa', 'acre', 'natal', 'belém',
                       'curitiba', 'imagem', 'texto', 'fortalezar', 'foto', 'belémcontracoronavírus'])

    print("\nExecution time - first part: %s seconds" % (time.time() - start_time))
    print("\nInitiating second part...")
    # ------- PART 2 -----------------------------------------------------------------------------------#
    df = rws[['Post Created Date', 'lemmatized']]
    df['lemma_str'] = [' '.join(map(str, l)) for l in df['lemmatized']]

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
    no_top_words = 30
    print("Printing topics... \n")
    display_topics(lda_model, tf_feature_names, no_top_words)

    panel = pyLDAvis.sklearn.prepare(lda_model, transf_matrix, count_vectorizer, mds='tsne')

    if SAVE:
        pyLDAvis.save_html(panel, './data/lda_result_mine2.html')

    topic_list = [['Topic0', 'Corona vírus e demais tópicos', '9.2%'],
                  ['Topic1', 'Início da vacinação contra a covid-19 em idosos', '12.5%'],
                  ['Topic2', 'Notícias sobre coronavírus, órgãos de saúde e abertura de leitos em hospitais para pacientes infectados com a covid-19', '17.4%'],
                  ['Topic3', 'Aumento do número de casos confirmados e ocorrência de óbitos de pessoas '
                             'infectadas pelo corona vírus', '13.1%'],
                  ['Topic4', 'Alerta sobre a pandemia do novo corona vírus e medidas de prevenção contra '
                             'a doença para a população', '47.7%']]

    if SAVE:
        with open(path + 'topics.txt', 'w', encoding='utf-8') as f:
            for i in topic_list:
                # print('{} - {} - {}'.format(i[0], i[1], i[2]))
                f.write('{} - {} - {}'.format(i[0], i[1], i[2]))
                f.write('\n')
        f.close()

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

            df_graph = df_document_topic
            if SAVE:
                df_graph.to_csv(path + 'document_topic2.csv', index=False)

        # seaborn graph
        if (GENERATE_GRAPH):
            df_graph.set_index('post_created_date').plot(kind='bar', stacked=True,
                                                   color=['blue', 'green', 'grey', 'red', 'orange'])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                       ncol=5, fancybox=True, shadow=True)
            plt.subplots_adjust(bottom=0.20)
            plt.xlabel("Mês de criação")
            plt.ylabel("Probabilidade de cada Tópico")

            plt.xticks(rotation=45)
            if SAVE:
                print("saving graph...")
                plt.savefig('./data/plot_graph.png')
                print("graph Saved!")
            plt.show()

    print("\nExecution time: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
