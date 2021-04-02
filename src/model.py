import matplotlib.pyplot as plt
f = plt.gcf()
import pandas as pd
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

from xgboost import XGBClassifier
import xgboost as xgb

import sklearn.metrics as metrics
from sklearn.metrics import roc_curve
from sklearn import linear_model

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from scipy.stats import ks_2samp
from scipy.stats import epps_singleton_2samp
import numpy as np

import shap

from scipy.stats import uniform
from scipy.stats import ttest_ind

import statsmodels.api as sm

def plot_logit_ci(result, out_dir, name, p=0.01):
    a = result.pvalues[result.pvalues <= p]
    err_series = result.params - result.conf_int()[0]
    coef_df = pd.DataFrame({'coef': result.params.values[1:],
                            'err': err_series.values[1:],
                            'varname': err_series.index.values[1:]
                            })

    coef_df = coef_df[coef_df['varname'].isin(list(a.axes[0]))] # keep just vars with low pvalue
    fig, ax = plt.subplots()

    coef_df.plot(x='varname', y='coef', kind='barh',
                 ax=ax, color='none', linewidth=0,
                 xerr='err', legend=False)
    ax.set_ylabel('')
    ax.scatter(x=coef_df['coef'],
               marker='s', s=50,
               y=pd.np.arange(coef_df.shape[0]), color='black')
    ax.axvline(x=0, linestyle='-', color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir + "/" + str(name) + '_CI.png')
    plt.savefig(out_dir + "/" + str(name) + '_CI.pdf')
    fig.clf()
    del fig


def ks_test(X, y, p=0.05):
    vars = list(X)
    dict = {}
    selected_features = []
    for var in vars:
        y0 = y[X[var]== 0]
        y1 = y[X[var] == 1]
        if (len(y0)>15) and (len(y1)>15):
            ks = ks_2samp(y0 ,y1)
            pv = ks.pvalue
        else:
            pv = 1
        if pv <=p:
            selected_features.append(var)
        dict[var] = {'KS-pv':pv, 'Avg-1':np.mean(y1), 'Avg-0':np.mean(y0)}#, 'y0_mean':np.mean(y0), 'y1_mean':np.mean(y1)}
    df = pd.DataFrame.from_dict(dict).T
    return selected_features, df

def es_test(X, y, p=0.05):
    vars = list(X)
    dict = {}
    selected_features = []
    for var in vars:
        y0 = y[X[var]== 0]
        y1 = y[X[var] == 1]
        if (len(y0)>25) and (len(y1)>25):
            try:
                es = epps_singleton_2samp(y0,y1)
            except:
                es = ks_2samp(y0,y1)

            pv = es.pvalue
        else:
            pv = 1
        if pv <=p:
            selected_features.append(var)
        dict[var] = {'ES-pv':pv, 'Avg-1':np.mean(y1), 'Avg-0':np.mean(y0)}#, 'y0_mean':np.mean(y0), 'y1_mean':np.mean(y1)}
    df = pd.DataFrame.from_dict(dict).T
    return selected_features, df


def t_test(X,y, p=0.05):
    vars = list(X)
    dict = {}
    selected_features = []
    for var in vars:
        y0 = y[X[var]== 0]
        y1 = y[X[var] == 1]
        if (len(y0)>25) and (len(y1)>25):
            try:
                es = ttest_ind(y0,y1)
            except:
                es = ttest_ind(y0,y1)

            pv = es.pvalue
        else:
            pv = 1
        if pv <=p:
            selected_features.append(var)
        dict[var] = {'t-pv':pv, 'Avg-1':np.mean(y1), 'Avg-0':np.mean(y0)}#, 'y0_mean':np.mean(y0), 'y1_mean':np.mean(y1)}
    df = pd.DataFrame.from_dict(dict).T
    return selected_features, df

def get_p_values_logit(y_train,X_train, out_dir, name, p=0.15, plot_=True):
    logit_model = sm.Logit(y_train,X_train)
    try:
        result = logit_model.fit(maxiter=100, disp=False)
        params = result.params
        conf = result.conf_int(alpha=0.05)
        odds = conf.copy()
        conf['Coef'] = params
        odds['Odds Ratio'] = params
        conf.columns = ['2.5', '97.5', 'Coef']
        odds.columns = ['2.5 ', '97.5 ', 'Odds Ratio']

        df = pd.DataFrame(conf)
        df = pd.concat([df, result.pvalues], axis=1)
        df = pd.concat([df, np.exp(odds)], axis=1)
        df.rename(columns={0:'pvalue'}, inplace=True)


    except:
        print('\nWARNING:' + name + ': Singlar Matrix!! -> pvalues procedure ignored\n')
        return list(X_train)
    if plot_:
        plot_logit_ci(result, out_dir, name, p=p)
    df2 = result.pvalues
    df2 = df2[df2 <= p]
    return list(df2.index.values), df


def adjusted_classes(y_scores, t=0.5):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def find_best_thsh(probs, y):
    f1_best = 0
    thrs = 0
    for t in np.linspace(0, 1, 100):
        pred = adjusted_classes(probs, t=t)
        f1 = metrics.f1_score(y, pred , average='weighted')
        if f1 >= f1_best:
            f1_best = f1
            thrs = t
    print('optimal thrsh : ' + str(thrs))
    return thrs


def feature_elimination(xtrain, y_train,  out_dir, name, max_vars=300):
    rfc = LogisticRegression(penalty='l2', solver='saga', max_iter=5000)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10),  n_jobs=10, scoring='roc_auc')
    rfecv.fit(xtrain, y_train)

    n = max(int(round(len(rfecv.support_==True )/2.5,0)) , min(max_vars, len(rfecv.support_==True )))
    print('n' + str(n))
    features = list(xtrain)

    #feature_ranks = rfecv.ranking_
    #feature_ranks_with_idx = enumerate(feature_ranks)
    #sorted_ranks_with_idx = sorted(feature_ranks_with_idx, key=lambda x: x[1])
    #top_n_idx = [idx for idx, rnk in sorted_ranks_with_idx[:n]]
    #selected_features = [features[i] for i in top_n_idx]

    #print(selected_features)
    a = rfecv.estimator_.coef_[0]
    abs_l = list(map(abs, a))
    idx = np.argsort(abs_l)[-n:]
    selected_features = [features[i] for i in idx]
    #print(selected_features)

    xv = len(selected_features)
    plt.figure(figsize=(4,4))
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('AUC', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
    plt.axvline(x=xv, ls='--', color='k')
    plt.xlim(len(xtrain.T), 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir + "/" + str(name) + '_FE.pdf')
    plt.savefig(out_dir + "/" + str(name) + '_FE.png')

    return selected_features


def plot_ROC(fig, ax, out_dir, name):
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.plot([0, 1], [0, 1], color='k')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc=4)
    plt.tight_layout()
    fig.savefig(out_dir + "/" + str(name) + '.png')
    fig.savefig(out_dir +"/"+ str(name)+'.pdf')

def get_roc_curve(y_test, probs):
    fpr, tpr, _ = roc_curve(y_test, probs)
    return fpr, tpr

def get_metrics(y_test, probs, thrs):
    result = {}
    predictions = adjusted_classes(probs, t=thrs)
    result['Accuracy'] = metrics.accuracy_score(y_test, predictions)
    result['F1w'] = metrics.f1_score(y_test, predictions, average='weighted')
    result['AUC'] = metrics.roc_auc_score(y_test, probs)
    result['Precision'] = metrics.average_precision_score(y_test, predictions)
    result['Recall'] = metrics.recall_score(y_test, predictions)

    return result


def classify_xgboost(dftrain, dftest, y_train, y_test, name='exp', out_dir='', explainer=False):
    model = XGBClassifier(silent=False,
                          scale_pos_weight=1,
       #                   learning_rate=0.1,
                          #colsample_bytree = 0.4,
                         # subsample = 0.8,
                          objective='binary:logistic',
                          n_estimators=500,
                         # reg_alpha = 0.3,
                          max_depth=4,
                          #gamma=10
                          )

    model.fit(dftrain, y_train)
    #model = model.best_estimator_
    pred_train = model.predict_proba(dftrain)
    pred_train = pd.DataFrame(pred_train)
    pred_train = pred_train[1]
    y_pred = model.predict_proba(dftest)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred[1]

    if explainer==True:
        xgboost_shap_explainer(model, dftrain, out_dir, name)

    #plt.figure(figsize=(4, 4))
    xgb.plot_importance(model)
    plt.title('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(out_dir+'/'+ name+ '_f-score.pdf')
    plt.savefig(out_dir + '/' + name + '_f-score.png')
    plt.close()

    coeff =  model.feature_importances_
    coeffs = {}
    i=0
    for col in list(dftrain):
        coeffs[col] = [coeff[i]]
        i += 1
    coeffs = pd.DataFrame.from_dict(coeffs)

    return y_test, y_pred, coeffs, pred_train


def classify_sklearn(dftrain, dftest, y_train, y_test, method):
    if method =='skl-SVM-l1':
        clf1 = LinearSVC(penalty='l1',dual=False, max_iter=10000)
        clf = CalibratedClassifierCV(clf1, cv=StratifiedKFold(10))

        clf.fit(dftrain, y_train)

        coef_avg = 0
        b = 0
        for i in clf.calibrated_classifiers_:
            a = i.predict_proba(dftrain)

            x = metrics.roc_auc_score(y_train, [j[1] for j in a])
            #print(b)
            if x>b:
                b = x
                clf_b = i.base_estimator
            #coef_avg = coef_avg + i.base_estimator.coef_

        #coeff = coef_avg / len(clf.calibrated_classifiers_)
        #clf = clf_b
        coeff = clf_b.coef_
        i = 0
        coeffs = {}
        for col in list(dftrain):
            coeffs[col] = [coeff[0][i]]
            i += 1
        coeffs = pd.DataFrame.from_dict(coeffs)

    if method == 'skl-LR-l1':
        clf = linear_model.LogisticRegression(penalty='l1',
                                                dual=False,
                                                solver='saga',
                                                max_iter=10000)
        clf = CalibratedClassifierCV(clf, cv=StratifiedKFold(15))

        clf.fit(dftrain, y_train)
        b = 0
        for i in clf.calibrated_classifiers_:
            a = i.predict_proba(dftrain)

            x = metrics.roc_auc_score(y_train, [j[1] for j in a])
            if x>b:
                b = x
                clf_b = i.base_estimator
        clf = clf_b
        coeff = clf.coef_
        i = 0
        coeffs = {}
        for col in list(dftrain):
            coeffs[col] = [coeff[0][i]]
            i += 1
        coeffs = pd.DataFrame.from_dict(coeffs)


    if method == 'skl-LR-l2':
        clf = linear_model.LogisticRegression(penalty='l2',
                                              #  dual=False,
                                              #  solver='saga',
                                                max_iter=10000)
        distributions = dict(C=uniform(loc=0, scale=4))
        clf = RandomizedSearchCV(clf, 
                                distributions, 
                                n_iter = 10,
                                random_state=0, 
                                cv=StratifiedKFold(10),
                                n_jobs=10)
        clf.fit(dftrain, y_train)
        clf = clf.best_estimator_
        coeff = clf.coef_
        i = 0
        coeffs = {}
        for col in list(dftrain):
            coeffs[col] = [coeff[0][i]]
            i += 1
        coeffs = pd.DataFrame.from_dict(coeffs)

    if method == 'skl-RF':
        clf = RandomForestClassifier()
        distributions = {'n_estimators': [50, 100, 150, 200, 300],
                       'max_depth': [5, 7, 10, 20, 30, 40, 50, 60],
                       'criterion': ['gini', 'entropy'],
                       }
        clf = RandomizedSearchCV(clf, 
                                distributions, 
                                n_iter = 2,
                                random_state=0, 
                                cv=StratifiedKFold(5), 
                                n_jobs=12)  
        clf.fit(dftrain, y_train)
        clf = clf.best_estimator_      
        coeffs = {}

    pred_train =  [i[1] for i in clf.predict_proba(dftrain)]
    y_pred = [i[1] for i in clf.predict_proba(dftest)]
    return y_test, y_pred, coeffs, pred_train


def xgboost_shap_explainer(model, X, out_dir, name):
    #plt.clf()
    fig, ax = plt.subplots()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)
    plt.tight_layout()
    fig.savefig(out_dir + "/" + str(name) + '_shap_forceplot.pdf')
    fig.savefig(out_dir + "/" + str(name) + '_shap_forceplot.png')
    fig.clf()
    #shap.force_plot(explainer.expected_value, shap_values, X, matplotlib=True)
    #shap.dependence_plot("RM", shap_values, X)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    fig.savefig(out_dir + "/" + str(name) + '_shap_summary.pdf')
    fig.savefig(out_dir + "/" + str(name) + '_shap_summary.png')
    fig.clf()
    return shap_values




def plot_ci(table, out_dir, name, p=0.01):
    #a = result.pvalues[result.pvalues <= p]
    #err_series = result.params - result.conf_int()[0]
    #coef_df = pd.DataFrame({'coef': result.params.values[1:],
    #                        'err': err_series.values[1:],
    #                        'varname': err_series.index.values[1:]
    #                        })

    fig, ax = plt.subplots()
    table['err'] = table['Odds Ratio'] - table['2.5 ']
    table = table[::-1]
    table.reset_index().plot(x='index', y='Odds Ratio', kind='barh',
                 ax=ax, color='none', linewidth=0,
                 xerr='err', legend=False
                 )
    ax.set_ylabel('')
    ax.scatter(x=table['Odds Ratio'],
               marker='s', s=50,
               y=pd.np.arange(table.shape[0]), color='black')
    ax.axvline(x=1, linestyle='-', color='black', linewidth=1)
    plt.tight_layout()

    plt.savefig(out_dir + "/" + str(name) + '_CI.pdf')
    plt.savefig(out_dir + "/" + str(name) + '_CI.png')

    fig.clf()

    del fig


def run_classification_models(xtrain, xtest, y_train, y_test, name='exp', out_dir='', max_steps=10000):

    selected_features, ks =  t_test(xtrain, y_train, p=0.07)
    pvalues, table = get_p_values_logit(y_train, xtrain, out_dir=out_dir, name=name, plot_=False)
    table = pd.concat([table, ks], axis=1)
    table.rename(columns={'p-value':'t-pv'},inplace=True)
    table = table[['Coef', '2.5', '97.5', 'Odds Ratio', '2.5 ', '97.5 ', 'Avg-0', 'Avg-1', 't-pv', 'pvalue']]
    table = table.round(3)
    #table.to_latex(out_dir + "/" + str(name) + '_vars.tex', column_format='lrrr|rrr|rrr|r')
    #table.to_csv(out_dir + "/" + str(name) + '_vars.csv')


    fig1, ax1 = plt.subplots(figsize=(4,4))
    df_result = {}
    coeffs_df = pd.DataFrame()


   # coeffs_df = pd.concat([coeffs_df, pvalues], axis=1)#, sort=True)

    for method in ['skl-SVM-l1', 'skl-LR-l1', 'skl-LR-l2', 'skl-RF', 'xgboost']: #, 'LC', 'Boosting']:#, 'CART', 'DNN', 'skl-SVM-l1']:
        if method == 'xgboost':
            y_test, probs, coeff, pred_train = classify_xgboost(dftrain= xtrain,
                                                    dftest=xtest,
                                                    y_train=y_train,
                                                    y_test=y_test,
                                                    name=name,
                                                    out_dir=out_dir,
                                                    explainer=True
                                                    )
        elif 'skl' in method:
            y_test, probs, coeff, pred_train = classify_sklearn(dftrain=xtrain,
                                                    dftest=xtest,
                                                    y_train=y_train,
                                                    y_test=y_test,
                                                    method=method)
        else:
            print('error in method')

        fpr, tpr = get_roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=method.replace('skl-',''))

        if method == 'skl-LR-l1':
            coeff = pd.DataFrame.from_dict(coeff).T
            coeff = coeff.rename(columns={0: 'LR-l1'})
            coeff = np.exp(coeff)
            coeffs_df = pd.concat([coeffs_df, coeff], axis=1, sort=True)

        elif method == 'skl-SVM-l1':
            coeff = pd.DataFrame.from_dict(coeff).T
            coeff = coeff.rename(columns={0: 'SVM-l1'})
            coeff = np.exp(coeff)
            coeffs_df = pd.concat([coeffs_df, coeff], axis=1, sort=True)

        elif method == 'skl-SVM-l2':
            coeff = pd.DataFrame.from_dict(coeff).T
            coeff = coeff.rename(columns={0: 'LR-l2'})
            coeff = np.exp(coeff)
            coeffs_df = pd.concat([coeffs_df, coeff], axis=1, sort=True)


        thrs = find_best_thsh(pred_train, y_train)
        #thrs = 0.5
        df_result[method] = get_metrics(y_test, probs, thrs)


    coeffs_df = coeffs_df.round(3)
    #coeffs_df = coeffs_df.sort_values(by=['LR-l1'], ascending = False)
    coeffs_df = coeffs_df.reindex(coeffs_df['LR-l1'].abs().sort_values().index)

    coeffs_df = coeffs_df.round(3)

    table = pd.concat([table, coeffs_df], axis=1)
    table = table.reindex(table.Coef.abs().sort_values().index).iloc[::-1]

    plot_ci(table, out_dir=out_dir, name=str(name), p=0.01)

    table.to_latex(out_dir + "/" + str(name) + '_vars.tex', column_format='lrrr|rrr|rrr|r')
    table.to_csv(out_dir + "/" + str(name) + '_vars.csv')
    coeffs_df.to_latex(out_dir + "/" + str(name) + '_coeffs.tex')
    coeffs_df.to_csv(out_dir + "/" + str(name) + '_coeffs.csv')

    df_result = pd.DataFrame(df_result)
    df_result.rename(columns={'skl-SVM-l1':'SVM-l1', 'skl-LR-l1':'LR-l1', 'skl-LR-l2':'LR-l2', 'skl-RF':'RF', 'xgboost':'XGBoost'} ,inplace=True)
    df_result = df_result.round(3)
    print(df_result)

    plot_ROC(fig1, ax1, out_dir, name)

    df_result.to_latex(out_dir +"/"+ str(name) + '.tex')
    df_result.to_csv(out_dir + "/" + str(name) + '.csv')
    plt.close('all')
    coeff_LC= {}


    return  df_result.T, coeff_LC
