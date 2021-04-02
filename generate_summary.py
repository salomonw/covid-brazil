import src.data_mgmt as dm
import src.model as m
import os
import pandas as pd
from src.utils import *
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

feature_dict = {'NU_NOTIFIC' : 'Registry ID',
               'DT_NOTIFIC' : 'Notification Date',
               'SEM_NOT' : 'Week Post Sympt',
               'DT_SIN_PRI' : 'Date First Sympt',
               'SEM_PRI' : 'Test Location ID',
               'SG_UF_NOT' : 'Test Location Federal',
               'ID_REGIONA' : 'Test Location Region ID',
               'CO_REGIONA' : 'Test Location Region',
               'ID_MUNICIP' : 'Test Location MunicipalityID',
               'CO_MUN_NOT' : 'Test Location Municipality',
               'ID_UNIDADE' : 'Test Unit ID',
               'CO_UNI_NOT' : 'Test Unit',
               'CS_SEXO' : 'Gender',
               'DT_NASC' : 'Birth Date',
               'NU_IDADE_N' : 'Age',
               'TP_IDADE' : 'Birth Date',
               'COD_IDADE' : 'Age code',
               'CS_GESTANT' : 'Gestational Age',
               'CS_RACA' : 'Race',
               'CS_ETINIA' : 'Indigenous',
               'CS_ESCOL_N' : 'Schooling',
               'ID_PAIS' : 'CountryID',
               'CO_PAIS' : 'CountryName',
               'SG_UF' : 'Residency ID',
               'ID_RG_RESI' : 'Residency Region ID',
               'CO_RG_RESI' : 'Residency Region',
               'ID_MN_RESI' : 'Residency Municipality',
               'CO_MUN_RES' : 'Residency Municipality ID',
               'CS_ZONA' : 'Residency Type',
               'SURTO_SG' : 'Acute Respiratory Distress Syndrome',
               'NOSOCOMIAL' : 'Contracted At Hospital',
               'AVE_SUINO' : 'Contact Birds Pigs',
               'FEBRE' : 'Fever',
               'TOSSE' : 'Cough',
               'GARGANTA' : 'Throat',
               'DISPNEIA' : 'Dyspnea',
               'DESC_RESP' : 'Respiratory Discomfort',
               'SATURACAO' : 'SpO2 less 95%',
               'DIARREIA' : 'Diarrhea',
               'VOMITO' : 'Vomiting',
               'OUTRO_SIN' : 'Other Symptoms',
               'OUTRO_DES' : 'Other Symptoms Description',
               'PUERPERA' : 'Postpartum',
               'CARDIOPATI' : 'Cardiovascular Disease',
               'HEMATOLOGI' : 'Hematologic Disease',
               'SIND_DOWN' : 'Down Syndrome',
               'HEPATICA' : 'Liver Chronic Disease',
               'ASMA' : 'Asthma',
               'DIABETES' : 'Diabetes',
               'NEUROLOGIC' : 'Neurological Disease',
               'PNEUMOPATI' : 'Another Chronic Pneumopathy',
               'IMUNODEPRE' : 'Immunosuppression',
               'RENAL' : 'Renal Chronic Disease',
               'OBESIDADE' : 'Obesity',
               'OBES_IMC' : 'BMI',
               'OUT_MORBI' : 'Other Risks',
               'MORB_DESC' : 'Other Risks Desc',
               'VACINA' : 'Flu Shot',
               'DT_UT_DOSE' : 'Flu Shot Date',
               'MAE_VAC' : 'Flu Shot Less 6 months',
               'DT_VAC_MAE' : 'Flu Shot Date Less 6 Months',
               'M_AMAMENTA' : 'Breast Feeds 6 Months',
               'DT_DOSEUNI' : 'Date Vaccine Children',
               'DT_1_DOSE' : 'Date Vaccine Children1',
               'DT_2_DOSE' : 'Date Vaccine Children2',
               'ANTIVIRAL' : 'Antiviral Use',
               'TP_ANTIVIR' : 'Type Antiviral',
               'OUT_ANTIV' : 'Type Antiviral Other',
               'DT_ANTIVIR' : 'Antiviral Start Date',
               'HOSPITAL' : 'Hospitalization',
               'DT_INTERNA' : 'Date Hospitalization',
               'SG_UF_INTE' : 'Hospital Region ID',
               'ID_RG_INTE' : 'Hospital Region IBGE2',
               'CO_RG_INTE' : 'Hospital Region IBGE',
               'ID_MN_INTE' : 'Hopspital MunicpialityID',
               'CO_MU_INTE' : 'Hopspital Municpiality',
               'ID_UN_INTE' : 'Hospital ID',
               'CO_UN_INTE' : 'Hospital ID',
               'UTI' : 'ICU',
               'DT_ENTUTI' : 'ICU start Date',
               'DT_SAIDUTI' : 'ICU end Date',
               'SUPORT_VEN' : 'Ventilator',
               'RAIOX_RES' : 'Xray Thorax Result',
               'RAIOX_OUT' : 'Xray Thorax Other',
               'DT_RAIOX' : 'Xray Test Date',
               'AMOSTRA' : 'Amostra',
               'DT_COLETA' : 'Amostra Date',
               'TP_AMOSTRA' : 'Amostra Type',
               'OUT_AMOST' : 'Amostra Other',
               'REQUI_GAL' : 'gal Sys Test',
               'IF_RESUL' : 'Test Result',
               'DT_IF' : 'Test Result Date',
               'POS_IF_FLU' : 'Test Influenza',
               'TP_FLU_IF' : 'Influenza Type',
               'POS_IF_OUT' : 'Positive Others',
               'IF_VSR' : 'Positive VSR',
               'IF_PARA1' : 'Positive Influenza 1',
               'IF_PARA2' : 'Positive Influenza 2',
               'IF_PARA3' : 'Positive Influenza 3',
               'IF_ADENO' : 'Positive Adenovirus',
               'IF_OUTRO' : 'Positive Other',
               'DS_IF_OUT' : 'Other Respiratory Virus',
               'LAB_IF' : 'Test Lab',
               'CO_LAB_IF' : 'Test Lab Other',
               'PCR_RESUL' : 'Result PCR',
               'DT_PCR' : 'Result PCR Date',
               'POS_PCRFLU' : 'Result PCR Influeza',
               'TP_FLU_PCR' : 'Result PCR Type Influeza',
               'PCR_FLUASU' : 'Result PCR SubType Influeza',
               'FLUASU_OUT' : 'Result PCR SubType Influeza_Other',
               'PCR_FLUBLI' : 'Result PCR SubType Influeza_Other_spec',
               'FLUBLI_OUT' : 'Result PCR SubType InfluezaB_Linage',
               'POS_PCROUT' : 'Result PCR Other',
               'PCR_VSR' : 'Result PCR VSR',
               'PCR_PARA1' : 'Result PCR parainfluenza1',
               'PCR_PARA2' : 'Result PCR parainfluenza2',
               'PCR_PARA3' : 'Result PCR parainfluenza3',
               'PCR_PARA4' : 'Result PCR parainfluenza4',
               'PCR_ADENO' : 'Result PCR adenovirus',
               'PCR_METAP' : 'Result PCR metapneumovirus',
               'PCR_BOCA' : 'Result PCR bocavirus',
               'PCR_RINO' : 'Result PCR rinovirus',
               'PCR_OUTRO' : 'Result PCR other',
               'DS_PCR_OUT' : 'Result PCR other name',
               'LAB_PCR' : 'Lab PCR',
               'CO_LAB_PCR' : 'LabP CR co',
               'CLASSI_FIN' : 'Result Final',
               'CLASSI_OUT' : 'Result Final other',
               'CRITERIO' : 'Result Final confirmation',
               'EVOLUCAO' : 'Evolution',
               'DT_EVOLUCA' : 'Death Date',
               'DT_ENCERRA' : 'Date Quarentine',
               'OBSERVA' : 'Other Observations',
               'DT_DIGITA' : 'Date Registry',
               'HISTO_VGM' : 'HISTO_VGM',
               'PAIS_VGM' : 'PAIS_VGM',
               'CO_PS_VGM' : 'CO_PS_VGM',
               'LO_PS_VGM' : 'LO_PS_VGM',
               'DT_VGM' : 'DT_VGM',
               'DT_RT_VGM' : 'DT_RT_VGM',
               'PCR_SARS2' : 'Result PCR Covid',
               'PAC_COCBO' : 'Occupation ID',
               'PAC_DSCBO' : 'Occupation Des'
              }
numeric_cols = ['Age',
           #'Gestational Age',
           'BMI'
                ]
date_cols = [
            'BirthDate',
           'ICUstartDate',
           'ICUendDate',
           'AmostraDate',
           'DeathDate',
           'DateQuarentine',]
region_var = 'Test Location Federal'
categorical_cols = ['Gender',
              'Race',
              #'Indigenous',
              'Schooling',
              'Acute Respiratory Distress Syndrome',
              'Contracted At Hospital',
              'Contact Birds Pigs',
              'Fever',
              'Cough',
              'Throat',
              'Dyspnea',
              'Respiratory Discomfort',
              'SpO2 less 95%',
              'Diarrhea',
              'Vomiting',
              'Other Symptoms',
              'Postpartum',
              'Cardiovascular Disease',
              'Hematologic Disease',
              'Down Syndrome',
              'Liver Chronic Disease',
              'Asthma',
              'Diabetes',
              'Neurological Disease',
              'Another Chronic Pneumopathy',
              'Immunosuppression',
              'Renal Chronic Disease',
              'Obesity',
              'Other Risks',
              #'Flu Shot Less 6 months',
              #'Breast Feeds 6 Months',
              'Antiviral Use',
              #'TypeAntiviral',
              'Hospitalization',
              'ICU',
              'Ventilator',
              'Xray Thorax Result',
              #'Amostra',
              'Result Final',
              'Evolution',
              #'OccupationID',
              #'Hospital ID',
              #'Hospital Region ID',
              #'Notification Date',
              region_var
              ]
keep_cols = categorical_cols.copy()
keep_cols.extend(numeric_cols)
post_hosp = [ 'Hospitalization',
              'Antiviral',
              'ICU',
              'Ventilator',
              'Xray Thorax Result',
              'Amostra',
              'Evolution'
              ]
post_death = [
              'Hospitalization',
              'Antiviral',
              'ICU',
              'Ventilator',
              'Amostra',
              'Evolution'
]

var_dictionary = {'typical':{'1.0':'', '2.0':' Nope '},
                  'Race':{'1.0':'White', '2.0':'Black', '3.0':'Yellow', '4.0':'Brown', '5.0':'Indigenous'},
                  'Schooling':{'0.0':'No Education', '1.0':'Elem 1-5', '2.0':'Elem 6-9', '3.0':'Medium 1-3', '4.0':'Superior', '5.0':'NA'},
                  'Xray Thorax Result':{'1.0':'Normal', '2.0':'Interstitial infiltrate', '3.0':'Consolidation', '4.0':'Mixed', '5.0':'Other','6.0':'Not done' },
                  'Ventilator':{'1.0':'Invasive', '2.0':'Non Invasive', '3.0':'No'},
                  'Evolution':{'1.0':'Recovered', '2.0':'Death'},
                  'Gender':{'M':'M', 'F':'F', 'I':'I'},
                  'Hospital':{'1.0':'Public', '2.0':'Private'},
                  'Region': {'DF' : 'Midwest',	'SP' : 'Southeast',	'SC' : 'South',	'RJ' : 'Southeast',
                                        'PR' : 'South',	'RS' : 'South',	'ES' : 'Southeast',	'GO' : 'Midwest',
                                        'MG' : 'Southeast',	'MS' : 'Midwest',	'MT' : 'Midwest',	'AP' : 'North',
                                        'RR' : 'North',	'TO' : 'North',	'RO' : 'North',	'RN' : 'Northeast',
                                        'CE' : 'Northeast',	'AM' : 'North',	'PE' : 'Northeast',	'SE' : 'Northeast',
                                        'AC' : 'North',	'BA' : 'Northeast',	'PB' : 'Northeast',	'PA' : 'North',
                                        'PI' : 'Northeast',	'MA' : 'Northeast',	'AL' : 'Northeast'}
                  }

#fname = 'data/bd_srag_08-06-2020.csv'
fname = 'data/INFLUD-30-06-2020.csv'
ts = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
outdir = 'results/' + ts
out_dir_da = outdir+'/Stats'
os.mkdir(outdir)
os.mkdir(out_dir_da)

dates = ['Date First Sympt', 'Date Hospitalization', 'ICU start Date', 'ICU end Date', 'Death Date']
keep_cols.extend(dates)


# Read Data
keep_cols.append('Date Registry')
df = dm.read_data(fname, feature_dict, keep_cols, sep=';')
df = dm.filter_positive_test(df)
df = dm.get_hosp_from_previous(df, fname='data/bd_srag_08-06-2020.csv',
                               feature_dict=feature_dict, keep_cols=keep_cols,
                                  categorical_cols=categorical_cols)
#df = df.sample(frac=0.05)





# Add 5 regions
df = df.rename(columns={region_var:'Region'})
categorical_cols.append('Region')
keep_cols.append('Region')
categorical_cols.remove(region_var)
keep_cols.remove(region_var)

categorical_cols.remove('Public Hospital')
categorical_cols.append('Hospital')


# Preprocessing
#TODO: what to do with jobs?
#table = dm.get_categorical_stats(df[categorical_cols], plot=True,  fname=out_dir_da+'/cat_stats')
df['Region'] = df['Region'].replace(var_dictionary['Region'])
print(df['Region'].unique())


df = df.drop(df[(df['Age'] > 45) & (df['Postpartum'] == 1)].index)


df = dm.set_mode_on_NA(df, numeric_cols)
df = dm.one_hot_encoding(df, categorical_features=categorical_cols)
df = dm.remove_features_containing(df, '9.0')
df = dm.var_to_categorical(df, var_name='Age', bins=[0,30,50,65,100])
df = dm.var_to_categorical(df, var_name='BMI', bins=[18.5,25,30,40,100])

df = dm.rename_features(df, var_dictionary, ignore_vars=['Age', 'Result Final', 'BMI', 'Gender', 'Region', 'Hospital'])

df['Race Brown/Black'] = df['Race Brown'] + df['Race Black']
df.drop(columns=['Race Brown'], inplace=True)
df.drop(columns=['Race Black'], inplace=True)

print(list(df))

df = dm.remove_features_containing(df, ' Nope ')
df = dm.remove_features_containing(df, 'NA')

df = df[(df['Evolution Death']==1) | (df['Evolution Recovered']==1)]


# Dates analysis
#dates = ['Date First Sympt', 'Date Hospitalization', 'ICU start Date', 'ICU end Date', 'Death Date']
for d in dates:
    df[d] = pd.to_datetime(df[d], errors='coerce')

df_die = df[df['Evolution Death']==1]
df_recovered = df[df['Evolution Recovered']==1]
df_vent = df[df['Ventilator Invasive']==1]

df_die["sym2die"] = df_die['Death Date'] -  df_die['Date First Sympt']
df_die["sym2die"] = pd.to_datetime(df_die["sym2die"], errors='coerce')
df_die["hosp2die"] = df_die['Death Date'] -  df_die['Date Hospitalization']


#df_recovered["hosp2die"] = df_recovered['Date Hospitalization'] -  df_recovered['Date First Sympt']

df_die["icu_time"] = df_die['ICU end Date'] -  df_die['ICU start Date']
df_recovered["icu_time"] = df_recovered['ICU end Date'] -  df_recovered['ICU start Date']

df_vent["vent_time"] = df_vent['ICU end Date'] -  df_vent['ICU start Date']

#df["sym2die"] = df['Death Date'] -  df['Date First Sympt']
#df["hosp2die"] = df['Date Hospitalization'] -  df['Date First Sympt']
df["icu_time"] = df['ICU end Date'] -  df['ICU start Date']
#df["vent_time"] = df['ICU end Date'] -  df['ICU start Date']


'''
print("-----")
print("sym2die")
print(df_die["sym2die"].mean())
print(df_die["sym2die"].std())
df_die["sym2die"].plot.hist()
df_die.hist(column="sym2die")
#plt.show()


print("-----")
print("hosp2die")
print(df_die["hosp2die"].mean())
print(df_die["hosp2die"].std())


print("-----")
print("ICU time for death patients")
print(df_die["icu_time"].mean())
print(df_die["icu_time"].std())


print("-----")
print("ICU time for death recovered")
print(df_recovered["icu_time"].mean())
print(df_recovered["icu_time"].std())

print("-----")
print("ICU time for all patients")
print(df["icu_time"].mean())
print(df["icu_time"].std())


print("-----")
print("Vent time for all patients")
print(df_vent["vent_time"].mean())
print(df_vent["vent_time"].std())
'''
#dfs = [df_die, df_recovered, df_vent]

#for v in list(df):
#    if "Race" in v:
#        df1 = df_die[df_die[v] == 1]
        #df2 = df_vent[df_die[v] == 1]
#        df3 = df[df[v] == 1]
#        print(v + '\t' + str(df1["sym2die"].mean()) + '('+str(df1["sym2die"].std())+')' +
#                '\t' + str(df1["hosp2die"].mean()) + '('+str(df1["hosp2die"].std())+')' )




obs = df.sum()
per = df.sum() / len(df)
s = pd.concat([obs, per], axis=1)
s[1] = "("+round((s[1] * 100),1).astype(str) + '%' +")"

s.to_latex(out_dir_da + '/summary.tex')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
  print(s.sort_index())





ages =[a for a in list(df) if 'Age' in a]
df2 = pd.DataFrame()
for age in ages:

  df2[age] = df[(df[age]==1) & (df['Evolution Death']==1)].sum() / df[df[age]==1].sum()

df2 = df2.T

for age in ages:
  del df2[age]

del df2['Evolution Death']
del df2['Evolution Recovered']
del df2['Result Final_5.0']
del df2['Gender_I']
del df2['Ventilator No']
del df2['Ventilator Non Invasive']

df2 = df2.T 
df2.sort_index(inplace=True)
plt.figure(figsize = (9, 13))
g = sns.heatmap(df2,
            annot=True,
            cmap=sns.cm.rocket_r,
            linewidth=1
            )
plt.tight_layout()
plt.savefig(out_dir_da + '/plt.png')
plt.savefig(out_dir_da + '/plt.pdf')


'''
rem_vars =['Xray Torax Result Normal',
           'Xray Torax Result Interstitial infiltrate',
           'Xray Torax Result Mixed',
           'Xray Torax Result Other',
           'Xray Torax Result Missing',
           #'Obesity ',
           'Result Final_5.0']

df.drop(columns=rem_vars, inplace=True)
#print(df.describe(include=['category']))

results = pd.DataFrame()
results  = df.T.index.copy()
print(df.sum())
results['n']  = df.sum()
#results['%']  = df.sum()/len(df)
print(results)

'''