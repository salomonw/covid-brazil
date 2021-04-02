# covid-brazil  (predictive models) 
Personalized models that predict mortality and the need for a mechanical ventilator for COVID-19 symptomatic patients.

## Publication:
- Wollenstein-Betech S, Silva A.A.B., Fleck J.L., Cassandras C.G., Paschalidis I.C.
***"Physiological and socioeconomic characteristics predict COVID-19 mortality and resource utilization in Brazil"*** (2020). PLOS ONE 15(10): e0240346. https://doi.org/10.1371/journal.pone.0240346

## Instructions:
## Install and use:
- To build local:
    - Requirements:
        - python3: 
            - numpy, pandas, matplotlib, seaborn, sklearn, tensorflow, xgboost, datetime.
    - To use:
            - run `python3 develop.py`
            - see results on the `results` directory
       
## Abstract
*Background:* 
Given the severity and scope of the current COVID-19 pandemic, it is critical to determine predictive features of COVID-19 mortality and medical resource usage to effectively inform health, risk-based physical distancing, and work accommodation policies. Non-clinical sociodemographic features are important explanatory variables of COVID-19 outcomes, revealing existing disparities in large health care systems.

*Methods and findings:* 
We use nation-wide multicenter data of COVID-19 patients in Brazil to predict mortality and ventilator usage. The dataset contains hospitalized patients who tested positive for COVID-19 and had either recovered or were deceased between March 1 and June 30, 2020. A total of 113,214 patients with 50,387 deceased, were included. Both interpretable (sparse versions of Logistic Regression and Support Vector Machines) and state-of-the-art non-interpretable (Gradient Boosted Decision Trees and Random Forest) classification methods are employed. Death from COVID-19 was strongly associated with demographics, socioeconomic factors, and comorbidities. Variables highly predictive of mortality included geographic location of the hospital (OR = 2.2 for Northeast region, OR = 2.1 for North region); renal (OR = 2.0) and liver (OR = 1.7) chronic disease; immunosuppression (OR = 1.7); obesity (OR = 1.7); neurological (OR = 1.6), cardiovascular (OR = 1.5), and hematologic (OR = 1.2) disease; diabetes (OR = 1.4); chronic pneumopathy (OR = 1.4); immunosuppression (OR = 1.3); respiratory symptoms, ranging from respiratory discomfort (OR = 1.4) and dyspnea (OR = 1.3) to oxygen saturation less than 95% (OR = 1.7); hospitalization in a public hospital (OR = 1.2); and self-reported patient illiteracy (OR = 1.1). Validation accuracies (AUC) for predicting mortality and ventilation need reach 79% and 70%, respectively, when using only pre-admission variables. Models that use post-admission disease progression information reach accuracies (AUC) of 86% and 87% for predicting mortality and ventilation use, respectively.

*Conclusions:* 
The results highlight the predictive power of socioeconomic information in assessing COVID-19 mortality and medical resource allocation, and shed light on existing disparities in the Brazilian health care system during the COVID-19 pandemic.
