import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(font="Times New Roman",font_scale=1.6)
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.set_style("ticks",{"ytick.minor.size":1})

df_final = pd.DataFrame(columns = ['si_sdr','sdr','sir','sar','stoi','instrument','method'])
for test in ['synth','real']:
    for model in ['ConvTasNet','UNet']:
        df = pd.read_csv(os.path.join('podcastmix','save',test+'_'+model+'_all_metrics.csv'))

        df_speech = pd.DataFrame(columns = ['si_sdr','sdr','sir','sar','stoi'])
        df_music = pd.DataFrame(columns = ['si_sdr','sdr','sir','sar','stoi'])
        for col in ['si_sdr','sdr','sir','sar','stoi']:
            df_speech[col]=pd.to_numeric(df[col].str.replace('[','').str.replace(']','').str.split().str[0])
            df_music[col]=pd.to_numeric(df[col].str.replace('[','').str.replace(']','').str.split().str[1])

        df_speech['source']='speech, PodcastMix-'+test
        df_music['source']='music, PodcastMix-'+test
        df_speech['method'] = model
        df_music['method'] = model
        df_final=pd.concat([df_final,df_speech,df_music])
df_final = df_final.rename(columns={'si_sdr': 'SI-SDR', 'sdr': 'SDR', 'sir': 'SIR', 'sar': 'SAR'})
df_final.sort_values('source', inplace=True, ascending=False)
df_plot = pd.melt(df_final, id_vars=['source','method'], value_vars=['SI-SDR','SDR','SIR','SAR'])
df_plot = df_plot.rename(columns={'method': 'model', 'variable': 'metric', 'value': 'dB'})


#import pdb;pdb.set_trace()
g = sns.catplot(
    data=df_plot, kind="bar", col="source",
    x="metric", y="dB", hue="model",
    ci="sd", palette="BuGn", alpha=.6, height=6,
    legend_out=False,
)
g.despine(left=True)

for i in range(4):
    # extract the matplotlib axes_subplot objects from the FacetGrid
    ax = g.facet_axis(0, i)

    # iterate through the axes containers
    for c in ax.containers:
        labels = [f'{(v.get_height()):.1f}' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge',color='g')
    ax.set_xlabel('')
plt.show()

