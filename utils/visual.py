import pandas as pd
from IPython.display import display_html
import matplotlib.pyplot as plt
import os

experiment_pref = 'experiment-log-'
test_file_pref = 'test_file_'
csv_ext = '.csv'
txt_ext = '.txt'


def display_synth_data(directory=None):
    container = ''

    if directory is None:
        directory = '/content/CIS-700/results/'

    for filename in os.listdir(directory):
        if filename.startswith(test_file_pref) and filename.endswith(txt_ext):
            fn_split = filename.split(test_file_pref)[1].split(txt_ext)[0].split('_')
            if len(fn_split) == 2:
                model = fn_split[0]
                training = fn_split[1]
                df = pd.read_csv(directory + filename, sep="\n", header=None)
                df.columns = [model + " " + training + " Synth Data"]
                df_styler = df.head(5).style.set_table_attributes("style='display:inline'")
                if container != '':
                    container += '<hr style="width: 900px; margin-left:0;">'
                container += df_styler._repr_html_()

    if container != '':
        file = open(directory + "real_synth_data.html", "w")
        file.write(container)
        file.close()
        display_html(container, raw=True)


def display_metrics(directory=None):

    df_list = []
    real_df_list = []
    oracle_df_list = []
    real_labels = []
    oracle_labels = []
    
    if directory is None:
        directory = '/content/CIS-700/results/'

    for filename in os.listdir(directory):
        if filename.startswith(experiment_pref) and filename.endswith(csv_ext):
            fn_split = filename.split(experiment_pref)[1].split(csv_ext)[0].split('-')
            if(len(fn_split) == 2):
                model = fn_split[0]
                training = fn_split[1]
                df = pd.read_csv(directory + filename)
                if training == 'real':
                    df = df.rename(columns={"EmbeddingSimilarity": "EmbSim_" + model.capitalize(), "nll-test": "Nll-Test_" + model.capitalize()})
                    real_df_list.append(df.set_index('epochs'))
                    real_labels.append(model)
                elif training == 'oracle':
                    df = df.rename(columns={"EmbeddingSimilarity": "EmbSim_" + model.capitalize(), "nll-test": "Nll-Test_" + model.capitalize(), "nll-oracle": "Nll-Oracle_" + model.capitalize()})
                    oracle_df_list.append(df.set_index('epochs'))
                    oracle_labels.append(model)
                #TODO Add CFG Training Logic Here

    real_results = pd.concat(real_df_list)
    oracle_results = pd.concat(oracle_df_list)

    # filter results to get separate lists for each metric type under each training
    filter_col = [col for col in real_results if col.startswith('EmbSim_') ]
    df_list.append(real_results[filter_col])
    filter_col = [col for col in real_results if col.startswith('Nll-Test')]
    df_list.append(real_results[filter_col])
    filter_col = [col for col in oracle_results if col.startswith('EmbSim_')]
    df_list.append(oracle_results[filter_col])
    filter_col = [col for col in oracle_results if col.startswith('Nll-Test')]
    df_list.append(oracle_results[filter_col])
    filter_col = [col for col in oracle_results if col.startswith('Nll-Oracle')]
    df_list.append(oracle_results[filter_col])

    # define number of rows and columns for subplots
    nrow = 3
    ncol = math.ceil(len(df_list) / nrow)

    # make a list of all dataframes
    df_title_list = ['Real EmbeddingSimilarites', 'Real NLL-Test', 'Oracle EmbeddingSimilarites', 'Oracle NLL-Test', 'Oracle NLL-Oracle']
    
    # plot counter
    count = 0
    
    #build plot
    fig, axes = plt.subplots(nrow, ncol)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for r in range(nrow):
        for c in range(ncol):
            if count < len(df_list):
                df = df_list[count]
                if df.columns.any('EmbSim_'):
                    df.columns = df.columns.str.replace(r'^EmbSim_', '')
                if df.columns.any('Nll-Test_'):
                    df.columns = df.columns.str.replace(r'^Nll-Test_', '')
                if df.columns.any('Nll-Oracle_'):
                    df.columns = df.columns.str.replace(r'^Nll-Oracle_', '')
                df.plot(ax=axes[r, c], y=df.columns, kind='line',
                            title=df_title_list[count], figsize=(20, 10))
                count += 1
    # save metrics to .png for later use in pdf report
    plt.savefig(directory + 'model_metric_charts.png')

