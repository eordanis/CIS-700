import pandas as pd
from IPython.display import display_html
import matplotlib.pyplot as plt
import os

directory = '/content/CIS-700/results' # update as needed
experiment_pref = 'experiment-log-'
test_file_pref = 'test_file_'
csv_ext = '.csv'
txt_ext = '.txt'


def display_synth_data():
    container = ''
    for filename in os.listdir(directory):
        if filename.startswith(test_file_pref) and filename.endswith(txt_ext):
            fn_split = filename.split(test_file_pref)[1].split(txt_ext)[0].split('_')
            if len(fn_split) == 2:
                model = fn_split[0]
                training = fn_split[1]
                df = pd.read_csv('results/' + filename, sep="\n", header=None)
                df.columns = [model + " " + training + " Synth Data"]
                df_styler = df.head(5).style.set_table_attributes("style='display:inline'")
                if container != '':
                    container += '<hr style="width: 900px; margin-left:0;">'
                container += df_styler._repr_html_()

    if container != '':
        file = open("results/real_synth_data.html", "w")
        file.write(container)
        file.close()
        display_html(container, raw=True)


def display_metrics():
    df_list = []
    labels = []
    for filename in os.listdir(directory):
        if filename.startswith(experiment_pref) and filename.endswith(csv_ext):
            fn_split = filename.split(experiment_pref)[1].split(csv_ext)[0].split('-')
            if len(fn_split) == 2:
                model = fn_split[0]
                training = fn_split[1]
                idx = 4
                if training == 'real':
                    idx = 3
                #in progress
                df = pd.read_csv('results/' + filename, sep="\n", header=None)  # .iloc[:, :idx]
                df_list.append(df)
                labels.append(model)

    # prep the metric dataframes
    oracle_sg = pd.read_csv('results/experiment-log-seqgan.csv').iloc[:, : 4]
    oracle_tg = pd.read_csv('results/experiment-log-textgan.csv').iloc[:, : 4]
    oracle_cg = pd.read_csv('results/experiment-log-cgan.csv').iloc[:, : 4]
    oracle_ig = pd.read_csv ('results/experiment-log-infogan.csv').iloc[:, : 4]
    #oracle_dg = pd.read_csv ('results/experiment-log-dcgan.csv').iloc[:, : 4]

    real_sg = pd.read_csv('results/experiment-log-seqgan-real.csv').iloc[:, : 3]
    real_tg = pd.read_csv('results/experiment-log-textgan-real.csv').iloc[:, : 3]
    real_cg = pd.read_csv('results/experiment-log-cgan-real.csv').iloc[:, : 3]
    real_ig = pd.read_csv ('results/experiment-log-infogan-real.csv').iloc[:, : 3]
    # real_dg = pd.read_csv ('results/experiment-log-dcgan-real.csv').iloc[:, : 3]

    # Create new dataframe to represent Oracle EmbeddingSimilarity across all models
    oracle_embed = pd.DataFrame({'Epochs': [1, 6, 10],
                                 'SeqGan': oracle_sg['EmbeddingSimilarity'],
                                 'TextGan': oracle_tg['EmbeddingSimilarity'],
                                 'CGan': oracle_cg['EmbeddingSimilarity'],
                                 'InfoGan': oracle_ig['EmbeddingSimilarity'],
                                 # 'DCGAN': oracle_dg['EmbeddingSimilarity'],
                                 })

    # Create new dataframe to represent Oracle nll-oracle across all models
    oracle_nll_orc = pd.DataFrame({'Epochs': [1, 6, 10],
                                   'SeqGan': oracle_sg['nll-oracle'],
                                   'TextGan': oracle_tg['nll-oracle'],
                                   'CGan': oracle_cg['nll-oracle'],
                                   'InfoGan': oracle_ig['nll-oracle'],
                                   # 'DCGAN': oracle_dg['nll-oracle'],
                                   })

    # Create new dataframe to represent Oracle nll-test across all models
    oracle_nll_test = pd.DataFrame({'Epochs': [1, 6, 10],
                                    'SeqGan': oracle_sg['nll-test'],
                                    'TextGan': oracle_tg['nll-test'],
                                    'CGan': oracle_cg['nll-test'],
                                    'InfoGan': oracle_ig['nll-test'],
                                    # 'DCGAN': oracle_dg['nll-test'],
                                    })

    # Create new dataframe to represent Real EmbeddingSimilarity across all models
    real_embed = pd.DataFrame({'Epochs': [1, 6, 10],
                               'SeqGan': real_sg[' EmbeddingSimilarity'],
                               'TextGan': real_tg['EmbeddingSimilarity'],
                               'CGan': real_cg['EmbeddingSimilarity'],
                               'InfoGan': real_ig['EmbeddingSimilarity'],
                               # 'LSGAN': real_lg['EmbeddingSimilarity'],
                               })

    # Create new dataframe to represent Real nll-test across all models
    real_nll_test = pd.DataFrame({'Epochs': [1, 6, 10],
                                  'SeqGan': real_sg['nll-test'],
                                  'TextGan': real_tg['nll-test'],
                                  'CGan': real_cg['nll-test'],
                                  'InfoGan': real_ig['nll-test'],
                                  # 'DCGAN': real_dg['nll-test'],
                                  })

    # define number of rows and columns for subplots
    nrow = 3
    ncol = 2
    # make a list of all dataframes
    df_list = [oracle_embed, oracle_nll_orc, oracle_nll_test, real_embed, real_nll_test]
    df_title_list = ['Oracle EmbeddingSimilarites', 'Oracle NLL-Oracle', 'Oracle NLL-Test',
                     'Real EmbeddingSimilarites',
                     'Real NLL-Test']
    # plot counter
    count = 0
    fig, axes = plt.subplots(nrow, ncol)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for r in range(nrow):
        for c in range(ncol):
            if count < 5:
                df_list[count].plot(ax=axes[r, c], x='Epochs', y=['SeqGan', 'TextGan', 'CGan', 'InfoGan'], kind='line',
                                    title=df_title_list[count], figsize=(20, 10))
                count += 1
    # save metrics to .png for later use in pdf report
    plt.savefig('results/model_metric_charts.png')

