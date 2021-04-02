import pandas as pd
import matplotlib.pyplot as plt


dir_1 = 'results/26-Jun-2020_01-34-53/PredictiveModels/'
dir_2 = 'results/04-Jul-2020_09-29-16/PredictiveModels/'


fname_1 = dir_1 + 'Death_0_vars.csv'
fname_2 = dir_2 + 'Death_0_vars.csv' #'Ventilator_vars.csv'

df1 = pd.read_csv(fname_1, index_col=0)
df2 = pd.read_csv(fname_2, index_col=0)

#print(df1)
print(df1.index)
df1 = df1.rename({'SpO2 less 95\% ':'SpO2 less 95% ', 'Xray Torax Result Consolidation ':'Xray Thorax Result Consolidation '}, axis='index')
#df1 = df1.replace('SpO2 less 95\%','SpO2 less 95%')#
print(df1)

def plot_ci(table, ax, colors=['black','red'], labels=False):
    #table.sort_index(inplace=True)
    table['err'] = table['Odds Ratio'] - table['2.5 ']
    table = table[::-1].reset_index()
    #print(table)
    table.plot(x='index', y='Odds Ratio', kind='barh',
                 ax=ax, color='none', linewidth=1,
                 xerr='err', ecolor='black', label='small'
                 )

    table.plot(x='index', y='Odds Ratio_2', kind='barh',
                 ax=ax, color='none', linewidth=1,
                 xerr='err_2', ecolor='red'
                 )

    ax.set_ylabel('')
    ax.scatter(x=table['Odds Ratio'],
               marker='s', s=50,
               y=pd.np.arange(table.shape[0]), color='black', label='small')

    ax.scatter(x=table['Odds Ratio_2'],
               marker='s', s=50,
               y=pd.np.arange(table.shape[0]), color='red', label='large')

    ax.axvline(x=1, linestyle='-', color='black', linewidth=1)


fig, ax = plt.subplots()

df2.columns = [str(col) + '_2' for col in df2.columns]
df = pd.concat([df2, df1], axis=1, sort=False)
print(df)

plot_ci(df, ax, labels=['large', 'small'])
#plot_ci(df1, ax[1], color= 'black', label='small')
#plt.show()
plt.tight_layout()
plt.show()

fig.clf()


