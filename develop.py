import src.data_mgmt as dm
import src.model as m
import os
import pandas as pd
from src.utils import *
from datetime import datetime

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
region_var = 'Test Location Federal'
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
           'DateQuarentine',
  ]
categorical_cols = ['Gender',
              'Race',
              #'Indigenous',
              'Schooling',
              'Acute Respiratory Distress Syndrome',
              'Contracted At Hospital',
              #'Contact Birds Pigs',
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
               #'Residency Region',
              #'OccupationID',
            #  'Hospital ID',
             # 'Hospital Region ID',
              #'Notification Date'
            region_var
              ]


keep_cols = categorical_cols.copy()
keep_cols.extend(numeric_cols)
post_hosp = [ 'Hospitalization',
              'Antiviral',
              'ICU',
              'Ventilator',
              'Xray Torax Result',
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

#fname = 'data/INFLUD-08-06-2020.csv'
fname = 'data/INFLUD-30-06-2020.csv'
ts = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
outdir = 'results/' + ts
out_dir_p = outdir+'/PredictiveModels'
out_dir_da = outdir+'/DataAnalytics'
os.mkdir(outdir)
os.mkdir(out_dir_p)
os.mkdir(out_dir_da)


# Read Data
keep_cols.append('Date Registry')
df = dm.read_data(fname, feature_dict, keep_cols, sep=';')
df = dm.filter_positive_test(df)
df = dm.get_hosp_from_previous(df, fname='data/INFLUD-08-06-2020.csv',
                               feature_dict=feature_dict, keep_cols=keep_cols,
                                  categorical_cols=categorical_cols)


'''

# Add public/private vari
df = dm.add_public_private_var(df, fname='data/ICU_beds.csv')
categorical_cols.append('Public Hospital')
keep_cols.append('Public Hospital')
df.drop(columns=['Hospital ID'], inplace=True )
categorical_cols.remove('Hospital ID')
keep_cols.remove('Hospital ID')
'''

# Add 5 regions
df = df.rename(columns={region_var:'Region'})
categorical_cols.append('Region')
keep_cols.append('Region')
categorical_cols.remove(region_var)
keep_cols.remove(region_var)

categorical_cols.remove('Public Hospital')
categorical_cols.append('Hospital')

# Add HDI
'''
fname = 'data/hdi.csv'
pp_df = pd.read_csv(fname)
pp_df.drop(columns=['public nonICU', 'private nonICU', 'public ICU', 'private ICU'], inplace=True)
df = df.merge(pp_df, how='left', on='Hospital ID')
del pp_df
df.rename(columns={'public hospital': 'Public Hospital'}, inplace=True)
df['Public Hospital'].loc[df['Public Hospital'] == 0] = 2.0
df['Public Hospital'].loc[df['Public Hospital'] == 1] = 1.0
df["Public Hospital"] = df["Public Hospital"].fillna(9.0)
'''


#TODO: CREATE CONGESTION METRIC

# Data Analytics
dm.create_basic_analytics(df, categorical_vars=categorical_cols, out_dir=out_dir_da)

# Preprocessing
#TODO: what to do with jobs?
#table = dm.get_categorical_stats(df[categorical_cols], plot=True,  fname=out_dir_da+'/cat_stats')
df['Region'] = df['Region'].replace(var_dictionary['Region'])

df = df.drop(df[(df['Age'] > 50) & (df['Postpartum'] == 1)].index)
df = dm.set_mode_on_NA(df, numeric_cols)
df = dm.one_hot_encoding(df, categorical_features=categorical_cols)
print(len(list(df)))
df = dm.remove_features_containing(df, '9.0')
print(len(list(df)))
df = dm.var_to_categorical(df, var_name='Age', bins=[0,30,50,65,100])

#df = dm.var_to_categorical(df, var_name='BMI', bins=[18.5,25,30,40,100])
df.drop(columns=['BMI'], inplace=True)

df = dm.rename_features(df, var_dictionary, ignore_vars=['Age', 'Result Final', 'BMI', 'Gender', 'Region'])

df['Race Brown/Black'] = df['Race Brown'] + df['Race Black']
df.drop(columns=['Race Brown'], inplace=True)
df.drop(columns=['Race Black'], inplace=True)


rem_vars =['Xray Thorax Result Normal',
           'Xray Thorax Result Interstitial infiltrate',
           'Xray Thorax Result Mixed',
           'Xray Thorax Result Other',
           'Xray Thorax Result Not done',
           # 'Obesity ',
           'Result Final_5.0']

print(len(list(df)))
df.drop(columns=rem_vars, inplace=True)

df = dm.remove_features_containing(df, ' Nope ')

print(len(list(df)))
df = dm.remove_features_containing(df, 'NA')
print(len(list(df)))

df = df[(df['Evolution Death']==1) | (df['Evolution Recovered']==1)]

df = dm.remove_corr_features(df, print_=False, plot_=False)
print(len(list(df)))

#dm.create_basic_analytics(df, categorical_vars=list(df), out_dir=out_dir_da)


# Classification
def run_model(df, name, y, remove_vars=False, max_vars=100):
    x_train, x_test, y_train, y_test = dm.create_training_and_test_sets(df, y=y, remove_vars=remove_vars,  percentage_train=0.7)
    selected_features, ktest_table = m.t_test(x_train, y_train, p=0.05)
    ktest_table['y'] = name
    x_train = x_train[selected_features]
    x_test = x_test[selected_features]
    print(len(list(x_train)))

    selected_features = m.feature_elimination(x_train, y_train, out_dir=out_dir_p, name=name, max_vars=max_vars)
    x_train = x_train[selected_features]
    x_test = x_test[selected_features]
    print(len(list(x_train)))


    m.run_classification_models(x_train, x_test, y_train, y_test, name=name, out_dir=out_dir_p, max_steps=10000)
    return ktest_table

# Run different classification models
#models = ['Hosp', 'Death', 'Death_hosp', 'ICU', 'Ventilator']#, 'XrayToraxResult']
models = [
           'Death_0','Death_1', 'Ventilator','Ventilator_small',
          'Ventilator_w_ICU', 'Ventilator_w_ICU_small',

          'Death_0_small', 'Death_1_small',
        #  'ICU', 'ICU_small',
]

ktest_g = pd.DataFrame(index=list(df))
for name in models:
    df0 = df.copy()
    df0 = df0[df0['Hospitalization '] == 1]
    max_vars = 100

    if name == 'Death_0' or name =='Death_0_small':
        y = 'Evolution Death'
        remove_vars = ['Antiviral', 'ICU',  'Ventilator', 'Evolution']
        if 'small' in name:
            max_vars = 10

    elif name == 'Death_1' or name =='Death_1_small':
        y = 'Evolution Death'
        remove_vars = ['Evolution']
        if 'small' in name:
            max_vars = 10

    elif name == 'ICU' or name=='ICU_small':
        y = 'ICU '
        remove_vars = ['ICU ', 'Ventilator', 'Evolution']
        if 'small' in name:
            max_vars = 10

    elif name == 'Ventilator' or name=='Ventilator_small':
        y = 'Ventilator Invasive'
        remove_vars = ['ICU ', 'Ventilator', 'Evolution']
        if 'small' in name:
            max_vars = 10

    elif name == 'Ventilator_w_ICU' or name=='Ventilator_w_ICU_small':
        y = 'Ventilator Invasive'
        remove_vars = ['Ventilator', 'Evolution']
        if 'small' in name:
            max_vars = 10

    print('\n--------------------\n'+name+'\n--------------------\n')
    ktest = run_model(df=df0, name=name, y=y, remove_vars=remove_vars,  max_vars=max_vars)
    #ktest_g = pd.concat([ktest_g, ktest], axis=1)


#ktest_g = ktest_g.round(decimals=2)
#ktest_g.to_latex(out_dir_da + '/pvTable.tex', column_format='lrrl|rrl|rrl|rrl|rrl' )

# Generate and save pdf report
shell('mv '+ out_dir_p + ' ' + 'results/report_template/media', printOut=False)
shell('mv '+ out_dir_da + ' ' + 'results/report_template/media', printOut=False)
os.chdir('results/report_template')
shell('pdflatex -interaction nonstopmode --jobname=report_'+ts+' main.tex', printOut=False)
shell('rm *.out *.log *.aux', printOut=False)
os.chdir('../..')
shell('mv results/report_template/report_'+ts+'.pdf  results/' + ts, printOut=False)
shell('mv results/report_template/media/PredictiveModels ' + 'results/' + ts , printOut=False)
shell('mv results/report_template/media/DataAnalytics ' + 'results/' + ts , printOut=False)