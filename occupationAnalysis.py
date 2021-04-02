import src.data_mgmt as dm
import src.model as m
import os
from src.utils import *
from datetime import datetime
import matplotlib.pyplot as plt

feature_dict = {'NU_NOTIFIC' : 'RegistryID',
               'DT_NOTIFIC' : 'NotificationDate',
               'SEM_NOT' : 'WeekPostSympt',
               'DT_SIN_PRI' : 'DateFirstSympt',
               'SEM_PRI' : 'TestLocationID',
               'SG_UF_NOT' : 'TestLocationFederal',
               'ID_REGIONA' : 'TestLocationRegionID',
               'CO_REGIONA' : 'TestLocationRegion',
               'ID_MUNICIP' : 'TestLocationMunicipalityID',
               'CO_MUN_NOT' : 'TestLocationMunicipality',
               'ID_UNIDADE' : 'TestUnitID',
               'CO_UNI_NOT' : 'TestUnit',
               'CS_SEXO' : 'Gender',
               'DT_NASC' : 'BirthDate',
               'NU_IDADE_N' : 'Age',
               'TP_IDADE' : 'BirthDate',
               'COD_IDADE' : 'Age_code',
               'CS_GESTANT' : 'GestationalAge',
               'CS_RACA' : 'Race',
               'CS_ETINIA' : 'Indigenous',
               'CS_ESCOL_N' : 'Schooling',
               'ID_PAIS' : 'CountryID',
               'CO_PAIS' : 'CountryName',
               'SG_UF' : 'ResidencyID',
               'ID_RG_RESI' : 'ResidencyRegionID',
               'CO_RG_RESI' : 'ResidencyRegion',
               'ID_MN_RESI' : 'ResidencyMunicipality',
               'CO_MUN_RES' : 'ResidencyMunicipalityID',
               'CS_ZONA' : 'ResidencyType',
               'SURTO_SG' : 'ComesFrom SG',
               'NOSOCOMIAL' : 'ContractedAtHospital',
               'AVE_SUINO' : 'ContactBirdsPigs',
               'FEBRE' : 'Fever',
               'TOSSE' : 'Cough',
               'GARGANTA' : 'Throat',
               'DISPNEIA' : 'Dyspneia',
               'DESC_RESP' : 'RespiratoryDisconfort',
               'SATURACAO' : 'Saturation',
               'DIARREIA' : 'Diarrhea',
               'VOMITO' : 'Vomiting',
               'OUTRO_SIN' : 'OtherSympthoms',
               'OUTRO_DES' : 'OtherSympthomsDescription',
               'PUERPERA' : 'Postpartum',
               'CARDIOPATI' : 'CardiovascularDisease',
               'HEMATOLOGI' : 'HematologicDisease',
               'SIND_DOWN' : 'DownSyndrome',
               'HEPATICA' : 'LiverChronicDisease',
               'ASMA' : 'Asthma',
               'DIABETES' : 'Diabetes',
               'NEUROLOGIC' : 'NeurologicDisease',
               'PNEUMOPATI' : 'AnotherChronicPneumatopathy',
               'IMUNODEPRE' : 'Immunosuppression',
               'RENAL' : 'RenalChronicDisease',
               'OBESIDADE' : 'Obesity',
               'OBES_IMC' : 'ObesityIMC',
               'OUT_MORBI' : 'OtherRisks',
               'MORB_DESC' : 'OtherRisksDesc',
               'VACINA' : 'FluShot',
               'DT_UT_DOSE' : 'FluShotDate',
               'MAE_VAC' : 'FluShotLess6months',
               'DT_VAC_MAE' : 'FluShotDateLess6Months',
               'M_AMAMENTA' : 'BreastFeeds6Months',
               'DT_DOSEUNI' : 'DateVaccineChildren',
               'DT_1_DOSE' : 'DateVaccineChildren1',
               'DT_2_DOSE' : 'DateVaccineChildren2',
               'ANTIVIRAL' : 'AntiviralUse',
               'TP_ANTIVIR' : 'TypeAntiviral',
               'OUT_ANTIV' : 'TypeAntiviralOther',
               'DT_ANTIVIR' : 'AntiviralStartDate',
               'HOSPITAL' : 'Hospitalization',
               'DT_INTERNA' : 'DateHospitalization',
               'SG_UF_INTE' : 'HospitalRegionID',
               'ID_RG_INTE' : 'HospitalRegionIBGE2',
               'CO_RG_INTE' : 'HospitalRegionIBGE',
               'ID_MN_INTE' : 'HopspitalMunicpialityID',
               'CO_MU_INTE' : 'HopspitalMunicpiality',
               'UTI' : 'ICU',
               'DT_ENTUTI' : 'ICUstartDate',
               'DT_SAIDUTI' : 'ICUendDate',
               'SUPORT_VEN' : 'Ventilator',
               'RAIOX_RES' : 'XrayToraxResult',
               'RAIOX_OUT' : 'XrayToraxOther',
               'DT_RAIOX' : 'XrayTestDate',
               'AMOSTRA' : 'Amostra',
               'DT_COLETA' : 'AmostraDate',
               'TP_AMOSTRA' : 'AmostraType',
               'OUT_AMOST' : 'AmostraOther',
               'REQUI_GAL' : 'galSysTest',
               'IF_RESUL' : 'TestResult',
               'DT_IF' : 'TestResultDate',
               'POS_IF_FLU' : 'TestInfluenza',
               'TP_FLU_IF' : 'InfluenzaType',
               'POS_IF_OUT' : 'PositiveOthers',
               'IF_VSR' : 'PostitiveVSR',
               'IF_PARA1' : 'PositiveInfluenza1',
               'IF_PARA2' : 'PositiveInfluenza2',
               'IF_PARA3' : 'PositiveInfluenza3',
               'IF_ADENO' : 'PositiveAdenovirus',
               'IF_OUTRO' : 'PositiveOther',
               'DS_IF_OUT' : 'OtherRespiratoryVirus',
               'LAB_IF' : 'TestLab',
               'CO_LAB_IF' : 'TestLabOther',
               'PCR_RESUL' : 'ResultPCR',
               'DT_PCR' : 'ResultPCR_Date',
               'POS_PCRFLU' : 'ResultPCR_Influeza',
               'TP_FLU_PCR' : 'ResultPCR_Type_Influeza',
               'PCR_FLUASU' : 'ResultPCR_SubType_Influeza',
               'FLUASU_OUT' : 'ResultPCR_SubType_Influeza_Other',
               'PCR_FLUBLI' : 'ResultPCR_SubType_Influeza_Other_spec',
               'FLUBLI_OUT' : 'ResultPCR_SubType_InfluezaB_Linage',
               'POS_PCROUT' : 'ResultPCR_Other',
               'PCR_VSR' : 'ResultPCR_VSR',
               'PCR_PARA1' : 'ResultPCR_parainfluenza1',
               'PCR_PARA2' : 'ResultPCR_parainfluenza2',
               'PCR_PARA3' : 'ResultPCR_parainfluenza3',
               'PCR_PARA4' : 'ResultPCR_parainfluenza4',
               'PCR_ADENO' : 'ResultPCR_adenovirus',
               'PCR_METAP' : 'ResultPCR_metapneumovirus',
               'PCR_BOCA' : 'ResultPCR_bocavirus',
               'PCR_RINO' : 'ResultPCR_rinovirus',
               'PCR_OUTRO' : 'ResultPCR_other',
               'DS_PCR_OUT' : 'ResultPCR_other_name',
               'LAB_PCR' : 'LabPCR',
               'CO_LAB_PCR' : 'LabPCR_co',
               'CLASSI_FIN' : 'ResultFinal',
               'CLASSI_OUT' : 'ResultFinal_other',
               'CRITERIO' : 'ResultFinal_confirmation',
               'EVOLUCAO' : 'Evolution',
               'DT_EVOLUCA' : 'DeathDate',
               'DT_ENCERRA' : 'DateQuarentine',
               'OBSERVA' : 'OtherObservations',
               'DT_DIGITA' : 'DateRegistry',
               'HISTO_VGM' : 'HISTO_VGM',
               'PAIS_VGM' : 'PAIS_VGM',
               'CO_PS_VGM' : 'CO_PS_VGM',
               'LO_PS_VGM' : 'LO_PS_VGM',
               'DT_VGM' : 'DT_VGM',
               'DT_RT_VGM' : 'DT_RT_VGM',
               'PCR_SARS2' : 'ResultPCR_Covid',
               'PAC_COCBO' : 'OccupationID',
               'PAC_DSCBO' : 'OccupationDes'
              }
numeric_cols = ['Age',
           'GestationalAge',
           'ObesityIMC'
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
              'ComesFrom SG',
              'ContractedAtHospital',
              'ContactBirdsPigs',
              'Fever',
              'Cough',
              'Throat',
              'Dyspneia',
              'RespiratoryDisconfort',
              'Saturation',
              'Diarrhea',
              'Vomiting',
              'OtherSympthoms',
              'Postpartum',
              'CardiovascularDisease',
              'HematologicDisease',
              'DownSyndrome',
              'LiverChronicDisease',
              'Asthma',
              'Diabetes',
              'NeurologicDisease',
              'AnotherChronicPneumatopathy',
              'Immunosuppression',
              'RenalChronicDisease',
              'Obesity',
              'OtherRisks',
              'FluShotLess6months',
              'BreastFeeds6Months',
              'AntiviralUse',
              'Hospitalization',
              'ICU',
              'Ventilator',
              'XrayToraxResult',
              'Amostra',
              'ResultFinal',
              'Evolution',
              'OccupationDes'
              ]
keep_cols = categorical_cols.copy()
keep_cols.extend(numeric_cols)
post_hosp = [ 'Hospitalization',
              'Antiviral',
              'ICU',
              'Ventilator',
              'XrayToraxResult',
              'Amostra',
              'Evolution'
              ]
post_death = [
              'Hospitalization',
              'Antiviral',
              'ICU',
              'Ventilator',
              'XrayToraxResult',
              'Amostra',
              'Evolution'
]

fname = 'data/bd_srag_04-05-2020.csv'
ts = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
#outdir = 'results/' + ts
#out_dir_p = outdir+'/PredictiveModels'
#out_dir_da = outdir+'/DataAnalytics'
#os.mkdir(outdir)
#os.mkdir(out_dir_p)
#os.mkdir(out_dir_da)


# Read Data
df = dm.read_data(fname, feature_dict, keep_cols)
df = dm.filter_positive_test(df)
#df = df.sample(frac=1)

# Histogram of Jobs
a = {i:len(df[df.OccupationDes==i]) for i in df.OccupationDes.unique() if str(i) !='nan'}
a = sorted(a.items(), key=lambda x:x[1], reverse=True)
x = [i[0] for i in a]
y = [i[1] for i in a]
print(sum(y))
#keys = a_dictionary.keys()
#values = a_dictionary.values()
plt.bar(x, y)
plt.show()
#df.OccupationDes.hist()
#df_hist = df.OccupationID.sort()
#df_hist.hist()
#plt.show()
# Data Analytics
#dm.create_basic_analytics(df, categorical_vars=categorical_cols, out_dir=out_dir_da)

# Preprocessing
#TODO: indigenous to categorical
#TODO: what to do with jobs?
#TODO: add numerical variables such as age and BMI
table = dm.get_categorical_stats(df[categorical_cols], plot=True,  fname=out_dir_da+'/cat_stats')
#df = dm.filter_positive_test(df)
df = dm.set_mode_on_NA(df, numeric_cols)
df = dm.one_hot_encoding(df, categorical_features=categorical_cols)
df = dm.remove_corr_features(df, print_=False)
df = dm.remove_9_features(df)

