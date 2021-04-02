import pandas as pd
import matplotlib.pyplot as plt


def plot_ci(table, ax, colors=['black','red'], labels=False):
    table['err'] = table['Odds Ratio'] - table['2.5 ']
    table = table[::-1].reset_index()
    table.plot(x='index', y='Odds Ratio', kind='barh',
                 ax=ax, color='none', linewidth=1,
                 xerr='err', ecolor='black'
                 )
    ax.set_ylabel('')
    ax.scatter(x=table['Odds Ratio'],
               marker='s', s=50,
               y=pd.np.arange(table.shape[0]), color='black')

    ax.axvline(x=1, linestyle='-', color='black', linewidth=1)
    ax.get_legend().remove()
    return  ax

dir = 'results/04-Jul-2020_09-29-16/PredictiveModels/'

fnames = []
fnames.append( dir + 'Death_1_vars')#.csv')
fnames.append( dir + 'Ventilator_w_ICU_vars')#.csv')

for fname in fnames:
    fig, ax = plt.subplots(figsize=(6,6))
    df = pd.read_csv(fname+".csv", index_col=0)
    ax = plot_ci(df, ax)
    plt.tight_layout()
    plt.savefig(fname+".pdf")
    #plt.show()
    fig.clf()

