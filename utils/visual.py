import pandas as pd
from IPython.display import display_html


def display_synth_data():
    # prep the synthetic text dataframes
    seqgan_data = pd.read_csv('results/seqgan_test_file.txt', sep="\n", header=None)
    seqgan_data.columns = ["SeqGAN Synth Data"]
    textgan_data = pd.read_csv('results/textgan_test_file.txt', sep="\n", header=None)
    textgan_data.columns = ["TextGAN Synth Data"]
    cgan_data = pd.read_csv('results/test_file_cgan.txt', sep="\n", header=None)
    cgan_data.columns = ["CGAN Synth Data"]
    # infogan_data = pd.read_csv('results/test_file_infogan.txt', sep="\n", header=None)
    # infogan_data.columns = ["INFOGAN Synth Data"]
    # lsgan_data = pd.read_csv('results/test_file_lsgan.txt', sep="\n", header=None)
    # lsgan_data.columns = ["LSGAN Synth Data"]

    # style synth data for inline display
    df1_styler = seqgan_data.head(5).style.set_table_attributes("style='display:inline'")
    df2_styler = textgan_data.head(5).style.set_table_attributes("style='display:inline'")
    df3_styler = cgan_data.head(5).style.set_table_attributes("style='display:inline'")
    # df4_styler = infogan_data.head(5).style.set_table_attributes("style='display:inline'")
    # df5_styler = lsgan_data.head(5).style.set_table_attributes("style='display:inline'")

    hrule = '<hr style="width: 900px; margin-left:0;">'
    container = df1_styler._repr_html_() + hrule + df2_styler._repr_html_() + hrule + df3_styler._repr_html_();
    file = open("results/synth_data.html", "w")
    file.write(container)
    file.close()
    display_html(container, raw=True)


def get_metric_df_lists():
    # prep the metric dataframes
    oracle_sg = pd.read_csv('results/experiment-log-seqgan.csv').iloc[:, : 4]
    oracle_tg = pd.read_csv('results/experiment-log-textgan.csv').iloc[:, : 4]
    oracle_cg = pd.read_csv('results/experiment-log-cgan.csv').iloc[:, : 4]
    # oracle_ig = pd.read_csv ('results/experiment-log-infogan.csv').iloc[:, : 4]
    # oracle_lg = pd.read_csv ('results/experiment-log-lsgan.csv').iloc[:, : 4]
    real_sg = pd.read_csv('results/experiment-log-seqgan-real.csv').iloc[:, : 3]
    real_tg = pd.read_csv('results/experiment-log-textgan-real.csv').iloc[:, : 3]
    real_cg = pd.read_csv('results/experiment-log-cgan-real.csv').iloc[:, : 3]
    # real_ig = pd.read_csv ('results/experiment-log-infogan-real.csv').iloc[:, : 3]
    # real_lg = pd.read_csv ('results/experiment-log-lsgan-real.csv').iloc[:, : 3]

    # Create new dataframe to represent Oracle EmbeddingSimilarity across all models
    oracle_embed = pd.DataFrame({'Epoch': [1, 6, 10],
                                 'SeqGan': oracle_sg['EmbeddingSimilarity'],
                                 'TextGan': oracle_tg['EmbeddingSimilarity'],
                                 'CGan': oracle_cg['EmbeddingSimilarity'],
                                 # 'INFOGAN': oracle_ig['EmbeddingSimilarity'],
                                 # 'LSGAN': oracle_lg['EmbeddingSimilarity'],
                                 })

    # Create new dataframe to represent Oracle nll-oracle across all models
    oracle_nll_orc = pd.DataFrame({'Epoch': [1, 6, 10],
                                   'SeqGan': oracle_sg['nll-oracle'],
                                   'TextGan': oracle_tg['nll-oracle'],
                                   'CGan': oracle_cg[' nll-oracle'],
                                   # 'INFOGAN': oracle_ig['nll-oracle'],
                                   # 'LSGAN': oracle_lg['nll-oracle'],
                                   })

    # Create new dataframe to represent Oracle nll-test across all models
    oracle_nll_test = pd.DataFrame({'Epoch': [1, 6, 10],
                                    'SeqGan': oracle_sg['nll-test'],
                                    'TextGan': oracle_tg['nll-test'],
                                    'CGan': oracle_cg['nll-test'],
                                    # 'INFOGAN': oracle_ig['nll-test'],
                                    # 'LSGAN': oracle_lg['nll-test'],
                                    })

    # Create new dataframe to represent Real EmbeddingSimilarity across all models
    real_embed = pd.DataFrame({'Epoch': [1, 6, 10],
                               'SeqGan': real_sg[' EmbeddingSimilarity'],
                               'TextGan': real_tg['EmbeddingSimilarity'],
                               'CGan': real_cg[' EmbeddingSimilarity'],
                               # 'INFOGAN': v_ig['EmbeddingSimilarity'],
                               # 'LSGAN': real_lg['EmbeddingSimilarity'],
                               })

    # Create new dataframe to represent Real nll-test across all models
    real_nll_test = pd.DataFrame({'Epoch': [1, 6, 10],
                                  'SeqGan': real_sg['nll-test'],
                                  'TextGan': real_tg['nll-test'],
                                  'CGan': real_cg['nll-test'],
                                  # 'INFOGAN': real_ig['nll-test'],
                                  # 'LSGAN': real_lg['nll-test'],
                                  })

    # make a list of all dataframes
    df_list = [oracle_embed, oracle_nll_orc, oracle_nll_test, real_embed, real_nll_test]
    df_title_list = ['Oracle EmbeddingSimilarites', 'Oracle NLL-Oracle', 'Oracle NLL-Test', 'Real EmbeddingSimilarites',
                     'Real NLL-Test']
    return {df_list, df_title_list}

