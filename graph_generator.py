import time
import pandas as pd
import matplotlib.pyplot as plt


GENERATE_GRAPH = False

def set_df_date(df):
    df['day'] = pd.to_datetime(df['date'])
    df['ano_mes'] = df['day'].dt.to_period('M')
    return df

def main():
    start_time = time.time()

    path = './data/'
    df_doc = pd.read_csv(path + 'document_topic.csv')
    df_cases = pd.read_csv(path + 'cases_treated.csv')

    df_cases = df_cases[['year-month', 'confirmed', 'deaths']].groupby('year-month').sum().reset_index()
    df_1st_row = pd.DataFrame([['2020-01', 0, 0]], columns=['year-month', 'confirmed', 'deaths'])
    df_cases = pd.concat([df_1st_row, df_cases], ignore_index=True)

    # seaborn graph
    GENERATE_GRAPH = True
    if (GENERATE_GRAPH):
        df_doc.set_index('post_created_date').plot(kind='bar', stacked=True,
                                               color=['Grey', 'Purple', 'Blue', 'Green', 'Orange'])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=5, fancybox=True, shadow=True)
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
        plt.savefig(path + 'plot_graph2.png')
        print("graph Saved!")
        plt.show()

    print("\nExecution time: %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()