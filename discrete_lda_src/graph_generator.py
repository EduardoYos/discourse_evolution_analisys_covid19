# script to generate graph based on number of cases and deaths of covid-19 and topics information
# extracted from city halls posts

# input file: "document_topic.csv" -> file with document topic data
# input file: "cases_treated.csv" -> file with treated cases data
# output file: "plot_graph2.png" -> image of generated graph with topics and cases information

import time
import pandas as pd
import matplotlib.pyplot as plt

GENERATE_GRAPH = False


def generate_topic_list(path):
    topic_list = [['Topic1', 'Corona vírus e demais tópicos', '9.2%'],
                  ['Topic2', 'Início da vacinação contra a covid-19 em idosos', '12.5%'],
                  ['Topic3',
                   'Notícias sobre coronavírus, órgãos de saúde e abertura de leitos em hospitais para pacientes infectados com a covid-19',
                   '17.4%'],
                  ['Topic4', 'Aumento do número de casos confirmados e ocorrência de óbitos de pessoas '
                             'infectadas pelo corona vírus', '13.1%'],
                  ['Topic5', 'Alerta sobre a pandemia do novo corona vírus e medidas de prevenção contra '
                             'a doença para a população', '47.7%']]

    with open(path + 'topics.txt', 'w', encoding='utf-8') as f:
        for i in topic_list:
            # print('{} - {} - {}'.format(i[0], i[1], i[2]))
            f.write('{} - {} - {}'.format(i[0], i[1], i[2]))
            f.write('\n')
    f.close()


def set_df_date(df):
    df['day'] = pd.to_datetime(df['date'])
    df['ano_mes'] = df['day'].dt.to_period('M')
    return df


def main():
    start_time = time.time()

    path = '../data/'
    results_path = '../results/'
    df_doc = pd.read_csv(path + 'document_topic_helper.csv')
    df_cases = pd.read_csv(path + 'cases_treated.csv')

    # generate_topic_list(results_path)

    df_doc['post_created_date'] = pd.to_datetime(df_doc['post_created_date'])
    df_doc['post_created_date'] = df_doc['post_created_date'].dt.strftime('%Y-%m')
    df_doc = df_doc.groupby('post_created_date').mean().reset_index()

    df_cases = df_cases[['year-month', 'confirmed', 'deaths']].groupby('year-month').sum().reset_index()
    df_1st_row = pd.DataFrame([['2020-01', 0, 0]], columns=['year-month', 'confirmed', 'deaths'])
    df_cases = pd.concat([df_1st_row, df_cases], ignore_index=True)

    df_cases = df_cases[(df_cases['year-month'] >= '2020-04') & (df_cases['year-month'] < '2021-01')].reset_index(
        drop=True)

    # seaborn graph
    GENERATE_GRAPH = True
    if (GENERATE_GRAPH):
        df_doc.rename(columns={'topic1': 'Casos (Resíduos)', 'topic2': 'Medidas de Proteção',
                               'topic3': 'Coronavírus', 'topic4': 'Saúde e Pandemia',
                               'topic5': 'Informações sobre Pandemia'}, inplace=True)

        df_doc.set_index('post_created_date').plot(kind='bar', stacked=True,
                                                   color=['Grey', 'Purple', 'Blue', 'Green', 'Orange'])

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                   ncol=2, fancybox=True, shadow=True, prop={'size': 8})
        plt.subplots_adjust(bottom=0.20, right=0.83)
        plt.xlabel("Mês de criação")
        plt.ylabel("Probabilidade de cada Tópico")

        plt.xticks(rotation=45)

        # creating second y axis
        y1_axis = plt.twiny()
        y2_axis = y1_axis.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'red'
        y2_axis.set_ylabel('Número de casos', color=color)  # we already handled the x-label with ax1
        y2_axis.plot(df_doc['post_created_date'], df_cases['confirmed'], color=color)
        y2_axis.get_xaxis().set_ticks([])
        y2_axis.locator_params('y', nbins=10)
        y2_axis.tick_params(axis='y', labelcolor=color)
        plt.ticklabel_format(style='plain', axis='y')

        print("saving graph...")
        plt.savefig(path + 'plot_graph_new.png')
        print("graph Saved!")
        plt.show()

    print("\nExecution time: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
