import time

import matplotlib
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
import warnings
import matplotlib.pyplot as plt
from IPython.core.display import HTML
from Cython import inline
from spacy.lang.pt.examples import sentences
from scipy.stats import entropy as scipy_entropy
#!python - m spacy download pt_core_news_sm

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
import plotly.express as px

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
#%matplotlib inline

import pkg_resources, imp
imp.reload(pkg_resources)
nlp = spacy.load("pt_core_news_sm")

# only when necessary
#nltk.download('stopwords')




# region Functions:

def add_date_columns(df):
    df['day'] = pd.to_datetime(df['Post Created Date'])
    df['ano_mes'] = df['day'].dt.to_period('M')
    df['week'] = df['day'].dt.week
    df['month'] = df['day'].dt.month
    df['year'] = df['day'].dt.year
    return df


def mayor2city(x, df):
    return df[df['mayor'] == x].city.values[0]


def pageName2city(x):
    return x.replace('do ', 'de ').partition(' de ')[2].lower()


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                                   token.pos_ in allowed_postags]))
    return texts_out


def filter_covid_words(df, words):
    len_total = len(df)
    df = df[df['Message'].str.contains("|".join(words), case=False, na=False)]
    len_covid = len_total - len(df)
    print('Foram removidos {} posts {}% do total'.format(len_covid, (len_covid/len_total)*100))
    return df


def dominant_topic(model, vec_data):
    lda_output = model.fit_transform(vec_data)
    return np.argmax(lda_output, axis=1)


def topic_distribution(df):
    #print(df.d_topic.value_counts()/df.d_topic.value_counts().sum())
    return (df.d_topic.value_counts()/df.d_topic.value_counts().sum())

# endregion


def main():
    start_time = time.time()
    warnings.filterwarnings("ignore")

    path = './data/'
    df_prefeituras = pd.read_csv(path + 'prefeituras.csv')
    df_depara = pd.read_csv(path + 'prefeito-cidade.csv')
    df_2 = pd.read_csv(path + 'prefeitos2020.csv')
    df_3 = pd.read_csv(path + 'prefeitos2021.csv')

    # df_prefeituras['city'] = df_prefeituras['Page Name'].apply(pageName2city)
    df_prefeituras['city'] = df_prefeituras['Page Name'].apply(pageName2city)

    df_2['city'] = df_2['Page Name'].apply(lambda x: mayor2city(x, df_depara))
    df_3['city'] = df_3['Page Name'].apply(lambda x: mayor2city(x, df_depara))

    df_prefeitos = pd.concat([df_2, df_3])
    df_final = pd.concat([df_prefeituras[['Message', 'city', 'Post Created Date']],
                          df_prefeitos[['Message', 'city', 'Post Created Date']]])

    df_final = add_date_columns(df_final)

    covid_words = ['comorbidade', 'ivermectina', 'tamiflu', 'azitromicina', 'lockdown', 'distanciamento social',
                   'hidroxicloroquina', 'covid19', 'isolamento', 'cloroquina', 'quarentena', 'covid_19', 'coronavirus',
                   'pandemia', 'corona', 'covid']

    df_final = filter_covid_words(df_final, covid_words)
    print(df_final['Message'][12])
    df_final['Message'] = df_final['Message'].str.replace(r'(?:http|https)\:\/\/[a-zA-Z0-9\.\/\?\:@\-_=#]+', '')
    print(df_final['Message'][12])

    #data visualization
    # df_timeline = df_final.groupby(['city', 'ano_mes']).size().reset_index()
    # df_timeline = df_timeline.groupby(['city', 'ano_mes']).sum().reset_index()
    # fig = px.line(x=df_timeline.iloc[:, 1].dt.strftime('%Y-%m'), y=df_timeline.iloc[:, 2], color=df_timeline.iloc[:, 0])
    # fig.update_layout(
    #     title='Post\'s por capital',
    #     xaxis_title="Data",
    #     yaxis_title="Quantidade"
    # )
    # fig.show()

    df_final['data_words'] = list(sent_to_words(df_final['Message'].values.tolist()))

    # !python - m spacy download pt_core_news_sm
    # import pkg_resources, imp
    # import spacy
    # imp.reload(pkg_resources)
    # nlp = spacy.load("pt_core_news_sm")

    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    df_final['lemmatization'] = lemmatization(df_final.data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(df_final['lemmatization'][:2])


    # vectorizer
    stop_words = nltk.corpus.stopwords.words('portuguese')
    stop_words.extend((list(df_final.city.unique())))
    stop_words.extend(['pracegover', 'macapa', 'imagem', 'texto', 'fortalezar'])
    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=5,  # minimum reqd occurences of a word
                                 stop_words=stop_words,  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 # max_features=50000,             # max number of uniq words
                                 )

    data_vectorized = vectorizer.fit_transform(df_final.lemmatization)

    # LDA Model
    lda_model = LatentDirichletAllocation(n_components=5,  # Number of topics
                                          max_iter=10,  # Max learning iterations
                                          learning_method='online',
                                          random_state=100,  # Random state
                                          batch_size=128,  # n docs in each learning iter
                                          evaluate_every=-1,  # compute perplexity every n iters, default: Don't
                                          n_jobs=-1,  # Use all available CPUs
                                          ).fit(data_vectorized)
    print(lda_model)  # Model attributes

    #lda_output = lda_model.fit_transform(data_vectorized)  # ?

    #pyLDAvis.enable_notebook()
    # print("prepare")
    panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
    #panel

    pyLDAvis.save_html(panel, './data/lda_result.html')

    #display(HTML('lda_result.html'))

    print("\n Execution time: %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
