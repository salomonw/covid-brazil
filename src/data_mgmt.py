
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#print(list(df))
#print(len(df))
#df['id'] = df.astype(str).values.sum(axis=1)
#print(len(df['id'].unique()))  # Registry ID


def read_data(fname, feature_dict=False, keep_cols=False, sep=','):
    df = pd.read_csv(fname, low_memory=False, sep=sep, error_bad_lines=False, encoding='latin-1')
    if feature_dict != False:
        df.rename(columns=feature_dict, inplace=True)
    if keep_cols != False:
        df = df[keep_cols]
    return df


def get_hosp_from_previous(df, fname='data/bd_srag_08-06-2020.csv',feature_dict=False, keep_cols=False, categorical_cols=False):
    # Read old dataset
    categorical_cols.append('Hospital ID')
    keep_cols.append('Hospital ID')
    df_old = read_data(fname, feature_dict, keep_cols, sep=';')
    df_old = filter_positive_test(df_old)
    df_old = add_public_private_var(df_old, fname='data/ICU_beds.csv')
    categorical_cols.append('Public Hospital')
    keep_cols.append('Public Hospital')
    df_old.drop(columns=['Hospital ID'], inplace=True)
    categorical_cols.remove('Hospital ID')

    # Create hash to compare
    a = ['Gender', 'Race', 'Schooling', 'Acute Respiratory Distress Syndrome', 'Contracted At Hospital', 'Fever',
         'Cough', 'Throat', 'Dyspnea', 'Respiratory Discomfort', 'SpO2 less 95%', 'Diarrhea', 'Vomiting',
         'Other Symptoms', 'Postpartum', 'Cardiovascular Disease', 'Hematologic Disease', 'Down Syndrome',
         'Liver Chronic Disease', 'Asthma', 'Diabetes', 'Neurological Disease', 'Another Chronic Pneumopathy',
         'Immunosuppression', 'Renal Chronic Disease', 'Obesity', 'Other Risks', 'Antiviral Use', 'Hospitalization',
         'ICU', 'Ventilator', 'Xray Thorax Result', 'Result Final']
    df['id'] = df[a].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    df_old['id'] = df_old[a].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # Sort
    df = df.sort_values(list(df))
    df_old = df_old.sort_values(list(df_old))

    # Merge /vlookup
    df = df.drop_duplicates(subset=['id'])
    df_old = df_old.drop_duplicates(subset=['id'])
    df_old = df_old[['id', 'Public Hospital']]
    df2 = df.merge(df_old, on='id', how='left')

    df2['Public Hospital'] = df2['Public Hospital'].fillna(9)
    df2.rename(columns={'Public Hospital':'Hospital'}, inplace=True)
    return df2

def get_var_stats(df, col):
    serie = pd.Series(df[col].values)
    v = pd.DataFrame(serie.value_counts(dropna=False, normalize=True), columns=[col])
    return v


def get_categorical_stats(df, plot=True, fname=''):
    df2 = pd.DataFrame()
    for col in list(df):
        v = get_var_stats(df, col)
        df2 = pd.concat([df2, v], axis=1, sort=False)
    df2.round(decimals=2)
    if plot:
        table2sns(df2.T, fname, show=False)
    return df2


def set_mode_on_NA(df, numeric_cols):
    for i in numeric_cols:
        df[i] = df[i].fillna(df[i].mode()[0])
    return df


def var_to_categorical(df, var_name, bins):
    for i in range(len(bins)-1):
        df[var_name + ' ' + str(bins[i]) + '-' + str(bins[i + 1])] = 0
        df[var_name + ' ' + str(bins[i]) + '-' + str(bins[i + 1])][(df[var_name]>bins[i])&(df[var_name]<=bins[i+1]) ] = 1
        if sum(df[var_name + ' ' + str(bins[i]) + '-' + str(bins[i + 1])])==0:
            df.drop(var_name + ' ' + str(bins[i]) + '-' + str(bins[i + 1]), axis=1, inplace=True)
    df.drop(var_name, axis=1, inplace=True)
    return df

def table2sns(df, fname, show=False, round=2):
    df.round(decimals=round)
    n = len(df)
    m = len(df.T)
    plt.figure(figsize = (m, n/3))
    g = sns.heatmap(df,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(fname+'.png')
    plt.savefig(fname+'.pdf')


def one_hot_encoding(df, categorical_features):
    for feature in categorical_features:
        dummies = pd.get_dummies(df[feature], drop_first=False, prefix=feature)
        df = pd.concat([df, dummies], axis=1)
        df.drop(feature, axis=1, inplace=True)
        #col_d = {i: i.split('_1')[0] for i in list(df)}
        #df = df.rename(columns=col_d)
    return df


def remove_corr_features(df, print_=False, plot_=False):
    correlated_features = set()
    correlated_matrix = df.corr()
    if plot_:
        plt.matshow(correlated_matrix)
        plt.show()
    for i in range(len(correlated_matrix.columns)):
        for j in range(i):
            if abs(correlated_matrix.iloc[i, j]) > 0.8:
                colname = correlated_matrix.columns[i]
                correlated_features.add(colname)
    for i in correlated_features:
        if 'Death' not in i:
            df.drop(i, axis=1, inplace=True)
    if print_:
        print('correlated vars: ' + str(correlated_features))
    return df


def remove_features_containing(df, text):
    features = list(df)
    features2 = []
    for feature in features:
        if  text not in feature:
            features2.append(feature)
    return df[features2]


def remove_NA_features(df):
    features = list(df)
    features2 = []
    for feature in features:
        if 'Na' not in feature:
            features2.append(feature)
    return df[features2]


def filter_positive_test(df):
    return df[df['Result Final']==5]

def create_training_and_test_sets(df, y, remove_vars, percentage_train):
    df0 = df.copy()
    for i in remove_vars:
        for  l in [s for s in list(df0) if i in str(s)]:
            if l != y :
                df0 = df0.drop(columns=[l])
    x_train = df0.sample(frac=percentage_train)
    x_test = pd.concat([df0, x_train]).drop_duplicates(keep=False)
    y_train = x_train.pop(y)
    y_test = x_test.pop(y)
    return x_train, x_test, y_train, y_test




def rename_features(df, var_dictionary, ignore_vars):
    sp_vars = list(var_dictionary.keys())
    for feature in list(df):
        f = feature.split('_')
        if len(f) > 1:
            if f[0] not in ignore_vars:
                if f[0] in sp_vars:
                    df = df.rename(columns={feature:f[0]+' '+var_dictionary[f[0]][f[1]]})
                else:
                    df = df.rename(columns={feature:f[0]+' '+var_dictionary['typical'][f[1]]})
    return df



def add_public_private_var(df, fname='data/ICU_beds.csv'):
    pp_df = pd.read_csv(fname)
    pp_df.drop(columns=['public nonICU', 'private nonICU', 'public ICU', 'private ICU'], inplace=True)
    df = df.merge(pp_df, how='left', on='Hospital ID')
    del pp_df
    df.rename(columns={'public hospital':'Public Hospital'}, inplace=True)
    df['Public Hospital'].loc[df['Public Hospital'] == 0] = 2.0
    df['Public Hospital'].loc[df['Public Hospital'] == 1] = 1.0
    df["Public Hospital"] = df["Public Hospital"].fillna(9.0)
    return df

def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    results['determination'] = ssreg / sstot

    return results



def create_summary(df, categorical_vars, out_dir):
    summ = {}
    n = len(df.Age)
    n_hosp = len(df[df.Hospitalization==1])
    n_death = len(df[df.Evolution==2])
    summ['N'] = [n]
    summ['N_Hosp'] = [n_hosp]
    summ['N_Death'] = n_death
    for feature in categorical_vars:
        if feature == 'Gender':
            Ni = len(df[df[feature] == 'F'])
            summ['Female'] = [Ni]
            Ni = len(df[df[feature] == 'M'])
            summ['Male'] = [Ni]
        #else:
        #    for cat in df.feature.unique():
         #       Ni = len(df[df[feature] == cat])
         #       summ[feature +'-' + cat] = [Ni]
        else:
            Ni = len(df[df[feature] == 1])
            summ[feature] = [Ni]
    print(pd.DataFrame.from_dict(summ).T)
    pd.DataFrame.from_dict(summ).T.to_latex(out_dir+'/summary.tex')
    pd.DataFrame.from_dict(summ).T.to_csv(out_dir+'/summary.csv' )






def create_age_hosp_plot(df, out_dir):
    N = len(df)
    fig, ax = plt.subplots(3, figsize=(7,8))
    age_list = []
    percentage_death_list = []
    ages = list(np.linspace(0,100,51))
    for j in range(len(ages)-1):# sorted(list(df.EDAD.unique())):
        num = len(df[(df.Age >= ages[j]) & (df.Age < ages[j+1]) &(df.Hospitalization==1)])
        den = len(df[(df.Age >= ages[j]) & (df.Age < ages[j+1])])
        ax[0].bar(x=ages[j+1] , height=num/den, color='blue')
        plt.grid(True)
        ax[2].scatter(x=ages[j+1], y=den, marker='*', color='red')
        plt.grid(True)
        age_list.append(ages[j+1])

        # Death
        num = len(df[(df.Age >= ages[j]) & (df.Age < ages[j + 1]) & (df.Evolution == 2)])
        ax[1].bar(x=ages[j + 1], height=num / den, color='blue')
        percentage_death_list.append(num / den)
        plt.grid(True)

    ax[0].set(ylabel="Percentage being Hospitalized")

    ## plot wih linear regression :
    x = age_list[-35:]
    y = percentage_death_list[-35:]
    results = polyfit(x, y, 1)
    [m,b]= results['polynomial']
    r2 = results['determination']
    yp = np.polyval([m, b], x)
    ax[1].plot(x, yp, color='orangered', label=str(round(b,2))+'+' + str(round(m,4))+'*Age; R2='+str(round(r2,2)))
    ax[1].legend()
    ax[1].set(ylabel="Deaths")
    ax[2].set(ylabel="Counts")
    plt.xlabel('Age')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_age.png')
    plt.savefig(out_dir + '/hosp_age.pdf')
    plt.close('all')



def create_cat_table(df, categorical_vars, out_dir):
    #print(categorical_vars)
    features = categorical_vars
    #features.remove('Evolution')
    #features.remove('Hospitalized')
    db1 = []
    db2 = []
    features_l = []
    special_features = ['Gender', 'Color', 'Race', 'Schooling', 'XrayToraxResult']
    for feature in features:
        if feature in special_features:
            for f in df[feature].unique():
                if f!=9.0:
                    Ni = len(df[df[feature] == f])
                    if Ni>10:
                        db1.append(len(df[(df[feature] == f) & (df['Hospitalization'] == 1)]) / Ni)
                        db2.append(len(df[(df[feature] == f) & (df['Evolution'] == 2)]) / Ni)
                        features_l.append(feature+'-'+str(f)+' (' + str(Ni) + ')')
                    else:
                        db1.append(np.NaN)
                        db2.append(np.NaN)
                        features_l.append(feature + ' (' + str(Ni) + ')')
        else:
            Ni = len(df[df[feature] == 1])
            if Ni>3:
                db1.append(len(df[(df[feature] == 1) & (df['Hospitalization'] == 1)]) / Ni)
                db2.append(len(df[(df[feature] == 1) & (df['Evolution'] == 2)]) / Ni)
                features_l.append(feature + ' (' + str(Ni) + ')')
            else:
                db1.append(np.NaN)
                db2.append(np.NaN)
                features_l.append(feature + ' (' + str(Ni) + ')')
    df_f = pd.DataFrame({'Death': db2}, index=features_l)
    df_f = df_f.sort_values('Death', ascending = False)
    df_f_transpose = df_f
    plt.figure(figsize = (8, 13))
    sns.heatmap(df_f_transpose, annot=True, cmap=sns.cm.rocket_r, linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_features.png')
    plt.savefig(out_dir + '/hosp_features.pdf')


def create_cat_age_table(df, categorical_vars, out_dir):
    dic = {}
    features = categorical_vars
    preconditions = ['Diabetes', 'Renal_Comorbidity', 'Cardio_Comorbidity', 'Lung_Comorbidity', 'Tobbaco-Use',
                     'Obesity', 'Respiratory distress']
    special_features = ['Gender', 'Color', 'Race', 'Schooling', 'XrayToraxResult']

    ages = np.arange(0, 105, 10)
    for i in range(len(ages)-1):
        df2 = df[(df.Age>ages[i]) & (df.Age<=ages[i+1])]
        d0 = df[(df.Age>ages[i]) & (df.Age<=ages[i+1])]
        db1 = []
        features_v = []
        for feature in features:
            if feature in special_features:
                for f in df[feature].unique():
                    if f !=9.0:
                        df2 = df[(df.Age>ages[i]) & (df.Age<=ages[i+1]) & (df[feature] == f)]
                        Ni = len(df2)
                        if Ni>=10:
                            db1.append(len(df2[(df2['Evolution'] == 2)]) / Ni)
                        else:
                            db1.append(np.nan)

                        features_v.append(feature + ' - ' + str(f))
            else:
                df2 = df[(df.Age>ages[i]) & (df.Age<=ages[i+1]) & (df[feature] == 1)]
                Ni = len(df2)
                if Ni >= 10:
                    db1.append(len(df2[df2['Evolution'] == 2]) / Ni)
                else:
                    db1.append(np.nan)
                if feature in preconditions:
                    d0 = d0[d0[feature] != 1]
                features_v.append(feature)

        N0 = len(d0)
        if N0 > 10:
            db1.append(len(d0[d0['Evolution'] == 2]) / N0)
        else:
            db1.append(np.nan)
        dic[str(ages[i])+'-'+str(ages[i+1])] = db1

    features_v.append('No preconditions')
    df_f = pd.DataFrame(dic, index=features_v)
    sns.set(font_scale=0.7)
    plt.figure(figsize=(15 * .7, 15 * .7))
    g = sns.heatmap(df_f,
                    annot=True,
                    cmap=sns.cm.rocket_r,
                    linewidth=1
                    )
    g.invert_yaxis()
    plt.xlabel('Age Group')

    plt.savefig(out_dir + '/hosp_age_features.png')
    plt.savefig(out_dir + '/hosp_age_features.pdf')
    '''
    #print(categorical_vars)
    features = categorical_vars
    #features.remove('Evolution')
    #features.remove('Hospitalized')
    db1 = []
    db2 = []
    features_l = []
    special_features = ['Gender', 'Color', 'Race', 'Schooling', 'XrayToraxResult']
    for feature in features:
        if feature in special_features:
            for f in df[feature].unique():
                Ni = len(df[df[feature] == f])
                if Ni>10:
                    db1.append(len(df[(df[feature] == f) & (df['Hospitalization'] == 1)]) / Ni)
                    db2.append(len(df[(df[feature] == f) & (df['Evolution'] == 2)]) / Ni)
                    features_l.append(feature+'-'+str(f)+' (' + str(Ni) + ')')
                else:
                    db1.append(np.NaN)
                    db2.append(np.NaN)
                    features_l.append(feature + ' (' + str(Ni) + ')')
        else:
            Ni = len(df[df[feature] == 1])
            if Ni>3:
                db1.append(len(df[(df[feature] == 1) & (df['Hospitalization'] == 1)]) / Ni)
                db2.append(len(df[(df[feature] == 1) & (df['Evolution'] == 2)]) / Ni)
                features_l.append(feature + ' (' + str(Ni) + ')')
            else:
                db1.append(np.NaN)
                db2.append(np.NaN)
                features_l.append(feature + ' (' + str(Ni) + ')')
    df_f = pd.DataFrame({'Death': db2}, index=features_l)
    df_f = df_f.sort_values('Death', ascending = False)
    df_f_transpose = df_f

    plt.figure(figsize = (8, 13))
    sns.heatmap(df_f_transpose, annot=True, cmap=sns.cm.rocket_r, linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir + '/hosp_features.png')
    plt.savefig(out_dir + '/hosp_features.pdf')
'''

def create_basic_analytics(df, categorical_vars, out_dir, type=1):
    # Build basic statistics table
    #features = list(df)
    #features.remove('Age')
    #features.remove('Evolution')
    #features.remove('Hospitalized')

    create_summary(df, categorical_vars, out_dir)
    create_age_hosp_plot(df, out_dir)
    create_cat_table(df, categorical_vars, out_dir)
    create_cat_age_table(df, categorical_vars, out_dir)



'''

    # Table Edad Test positive
    dic = {}
    preconditions = ['Diabetes', 'Renal_Comorbidity', 'Cardio_Comorbidity', 'Lung_Comorbidity', 'Tobbaco-Use', 'Obesity', 'Respiratory distress']
    for age_group in sorted(df.Age.unique()):
        d0 = df[(df.Age == age_group)]
        db1 = []
        features_v = []
        for feature in features:
            if feature == 'Gender':
                df2 = df[(df.Age == age_group) & (df[feature] == 1)]
                Ni = len(df2)
                if Ni > 5:
                    db1.append(len(df2[(df2['Hospitalized'] == 1)]) / Ni)
                else:
                    db1.append(np.nan)
                features_v.append(feature + '- Female')

                df2 = df[(df.Age == age_group) & (df[feature] == 0)]
                Ni = len(df2)
                if Ni > 5:
                    db1.append(len(df2[(df2['Hospitalized'] == 1)]) / Ni)
                else:
                    db1.append(np.nan)
                features_v.append(feature + '- Male')

            elif feature == 'Color':
                for color in df[feature].unique():
                    df2 = df[(df.Age == age_group) & (df[feature] == color)]
                    Ni = len(df2)
                    if Ni>5:
                        db1.append(len(df2[(df2['Hospitalized'] == 1)]) / Ni)
                    else:
                        db1.append(np.nan)
                    features_v.append(feature + ' - ' + color)

            else:
                df2 = df[(df.Age == age_group) & (df[feature] == 1)]
                Ni = len(df2)
                if Ni > 5:
                    db1.append(len(df2[df2['Hospitalized'] == 1]) / Ni)
                else:
                    db1.append(np.nan)
                if feature in preconditions:
                    d0 = d0[d0[feature] != 1]
                features_v.append(feature)

        N0 = len(d0)
        if N0 > 5:
            db1.append(len(d0[d0['Hospitalized'] == 1]) / N0)
        else:
            db1.append(np.nan)
        dic[age_group] = db1

    features_v.append('No preconditions')
    df_f = pd.DataFrame(dic, index=features_v)
    sns.set(font_scale=0.7)
    plt.figure(figsize = (15*.7,9*.7))
    g = sns.heatmap(df_f,
                annot=True,
                cmap=sns.cm.rocket_r,
                linewidth=1
                )
    g.invert_yaxis()
    plt.xlabel('Age Group')

    plt.savefig(out_dir + '/hosp_age_features.png')
    plt.savefig(out_dir + '/hosp_age_features.pdf')

'''