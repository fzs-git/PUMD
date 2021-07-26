import numpy as np
import time
import pandas as pd
import copy
import os
from xgboost import XGBClassifier
from PUMD import PUMD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.svm import SVC
import math
import matplotlib.pylab as plt
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, minmax_scale
import time
def model_performance(y_pred,y_true,y_prob=False):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(tn, fp, fn, tp)
    recall = tp / (tp + fn)  #tp/p
    specificity = tn / (tn + fp)   #tn/n
    precision = tp/(tp+fp)
    gmean = np.sqrt(recall * specificity)
    f_score = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)


    if np.array(y_prob).any():
        auc = roc_auc_score(y_true, y_prob)
        performance = [f_score, gmean, recall,precision, acc,auc,mcc]
        # fpr, tpr, thersholds = roc_curve(y_true, y_pred)

        # precision, recall, th = precision_recall_curve(y_true, y_prob)
        # f1s = 2 * precision[:-1] * recall[:-1]/(precision[:-1] + recall[:-1])
        # best_th = np.argmax(f1s)
        # print('best_fq:',np.max(f1s))

        # plt.figure(1)
        # plt.plot(fpr, tpr)
        # plt.xlabel('fpr')
        # plt.ylabel('tpr')
        # plt.show()


    else:
        performance = [f_score, gmean, recall, precision, acc,mcc]
    return np.array(performance)


def generate_feature_select(df,delete_list):
    # print(len(delete_list))
    df = df.drop(delete_list, axis=1)
    # print(len(df.columns))
    return df
def feature_select_func(np_org,delete_list,axis=1):
    np_fs = np.delete(np_org,delete_list,axis)
    print(len(np_fs))
    return np_fs

def generate_pudata(seed,data_path,label_frequency):
    rdg = np.random.RandomState(seed)
    df_all = pd.read_csv(data_path)
    # delete_feature = list(np.load(r'D:\Project\Python\data\new-4-6\Intersection_list_ex.npy'))
    # delete_feature = list(np.load(r'D:\Project\Python\data\new-4-6\Intersection_list.npy'))
    delete_feature = []
    df_all = generate_feature_select(df_all,delete_feature)
    columns_list = df_all.columns.to_list()
    feature_list = columns_list[1:len(columns_list)-1]

    df_mali = df_all[df_all['family_no'].isin(range(9))]
    mali_no = list(df_mali.family_no.value_counts().reset_index(name='counts')['index'])
    mali_count = list(df_mali.family_no.value_counts().reset_index(name='counts')['counts'])
    df_P = pd.DataFrame(columns=df_all.columns.to_list())
    #随机取label_frequency的P
    for no, num in zip(mali_no, mali_count):
        cut_num =int(np.ceil(num * label_frequency))
        #print("family_no", no, ":", cut_num)
        df_indices = pd.DataFrame(rdg.permutation(df_mali[df_mali['family_no']==no])[:cut_num], columns=columns_list )
        df_P = pd.concat([df_P,df_indices],axis=0)
    P_list = df_P.fqdn_no.to_list()
    df_U = df_all[~df_all['fqdn_no'].isin(P_list)]

    #dataFrame转numpy格式，并随机打乱顺序
    df_P = pd.DataFrame(rdg.permutation(df_P),columns=columns_list )
    data_P = np.array(df_P.iloc[:,1:len(feature_list)+1])
    label_P = np.array(df_P.family_no.to_list())

    df_U = pd.DataFrame(rdg.permutation(df_U),columns=columns_list )
    data_U = np.array(df_U.iloc[:,1:len(feature_list)+1])
    label_U = np.array(df_U.family_no.to_list())
    pred_U = []
    for c in  label_U :
        if c == 9:
            pred_U.append(0)
        else :
            pred_U.append(1)
    pred_U = np.array(pred_U)
    # print('P:',len(data_P))
    # print('U:',len(data_U))
    # print('benign:',len(df_all)-len(df_mali))
    # print('mali:',  len(df_mali))
    # print('P~:', len(df_mali)-len(data_P))
    # print(len(columns_list),len(feature_list),data_P.shape[1])

    return data_P,label_P,data_U,pred_U,label_U,df_P,df_U,feature_list

def generate_mix_freq_Train_data(mix_frequency, data_U, label_U, data_P):
    sum_mix = math.ceil(len(data_P) * mix_frequency)
    df_cnt = pd.Series(label_U).value_counts().rename_axis('unique_values').reset_index(name='counts').iloc[1:,:]
    df_cnt['mix_value'] = [int(math.ceil(i * sum_mix / np.sum(df_cnt['counts']))) for i in df_cnt['counts'].to_list()]
    df_cnt.iloc[0, 2] = int(sum_mix - np.sum(df_cnt['mix_value'].to_list()[1:]))
    # print(df_cnt)
    # print(sum_mix)
    y_train = np.empty([1,data_U.shape[1]])
    for i in range(9):
        i_bool = [True if x == i else False for x in label_U]
        tmp = df_cnt[df_cnt['unique_values'].isin([i])]['mix_value'].to_list()
        if len(tmp)<1:
            y_train = y_train
        else:
            tmp_data = [data_U[i]  for i , i_bool in enumerate(i_bool) if i_bool  ]
            slice_num = tmp[0]
            y_train = np.r_[y_train, tmp_data[:slice_num]]
    y_train = y_train[1:]
    # print(y_train.shape)
    j_bool = [True if x == 9 else False for x in label_U]
    y_train = np.r_[y_train, data_U[j_bool][:int(len(data_P) - sum_mix)]]
    rdg = np.random.RandomState(2021)
    df_y = pd.DataFrame(y_train)
    y_train = np.array(rdg.permutation(df_y))
    # print(y_train.shape)
    return y_train

def find_best_threshold(y_prob,pred_U):
    f1_list = []
    for i in y_prob:
        y_pred =  [1 if prob >= i else 0 for prob in y_prob]
        f_score = f1_score(pred_U, y_pred)
        f1_list.append(f_score)
    return np.max(f1_list) , y_prob[np.argmax(f1_list)]
def find_best_f1(y_prob,pred_U):
    df_pr = pd.DataFrame(columns=['y_prob','pred_U'])
    df_pr['y_prob']=y_prob
    df_pr['pred_U'] = pred_U
    df_pr.sort_values(by=["y_prob"],ascending=[False],inplace=True)
    f1_list = []

    for i in range(0, np.sum(pred_U) * 4):
        U_pred = np.r_[np.ones(i), np.zeros(len(pred_U) - i)]
        f1 = f1_score(df_pr['pred_U'].to_list(),U_pred)
        # mcc = matthews_corrcoef(df_pr['pred_U'].to_list(),U_pred)
        f1_list.append(f1)
        # mcc_list.append(mcc)
    # [f_score, gmean, recall, precision, acc, auc, mcc]
    U_pred = np.r_[np.ones(np.argmax(f1_list)), np.zeros(len(pred_U) - np.argmax(f1_list))]

    return model_performance(U_pred,df_pr['pred_U'].to_list(),df_pr['y_prob'].to_list())


# def generate_balance_pudata(seed,data_path,label_frequency,method = 'auto'):
#     #method = {'auto','smote',}

xgb = XGBClassifier(n_estimators=350, learning_rate=0.25, max_depth=6, n_jobs=-1, random_state=2018)
svm = SVC(C=1.0, kernel='linear', gamma='auto', probability=True, degree=3, random_state=2018)
rf = RandomForestClassifier(n_estimators=350, max_depth=6, random_state=2018)
lr = LogisticRegression(penalty='l2', C=1.0, solver='sag', max_iter=1000, random_state=2019, n_jobs=-1)
# solver lbfgs

def performance_contrast(seed,data_path,label_frequency):
    #[f_score, gmean, recall, precision, acc, auc, mcc]
    df_value = []
    df_f_score,df_gmean,df_recall,df_precision,df_acc,df_auc,df_mcc = [label_frequency],[label_frequency],[label_frequency],[label_frequency],[label_frequency],[label_frequency],[label_frequency]
    start = time.time()
    data_P,label_P,data_U,pred_U,label_U,df_P,df_U,feature_list = generate_pudata(seed,data_path,label_frequency)


    #-----------------------------------method_evaluate--------------------------------------
    mode = 'run'
    print('label_frequency:',label_frequency)
    ## [1]PUMD-RF
    ### RF
    lf_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    beta_list = [0.9,1.0,1.0,1.1,1.0,1.1,1.3,1.3,1.9]
    beta_dict = dict(zip(lf_list,beta_list))
    pumd_rf = PUMD(data_P, data_U, rf, return_proba=True, pred_U=pred_U, feature_list=feature_list, mode=mode ,ad_para=19,contamination=0.046,ad_beta=beta_dict[label_frequency])
    y_pred, y_prob = pumd_rf.predict()
    print(len(y_pred))
    pumd_rf_performance = model_performance(y_pred, pred_U, y_prob)
    print('pumd_rf_performance_org:', pumd_rf_performance)
    df_f_score.append(pumd_rf_performance[0])
    df_gmean.append(pumd_rf_performance[1])
    df_recall.append(pumd_rf_performance[2])
    df_precision.append(pumd_rf_performance[3])
    df_acc.append(pumd_rf_performance[4])
    df_auc.append(pumd_rf_performance[5])
    df_mcc.append(pumd_rf_performance[6])



    ##预处理
    dataset_scaled = StandardScaler().fit_transform(np.r_[data_P, data_U])
    data_P = dataset_scaled[:len(data_P)]
    data_U = dataset_scaled[len(data_P):]
    ## [2]PU_Biased-learning
    ### RF
    # # # v1直接计算 cost_fn和cost_fp
    cost_fn = len(data_U)/len(data_P)+len(data_U)
    cost_fp = len(data_P)/len(data_U)+len(data_P)
    X_train = np.r_[data_P, data_U]
    y_train = np.r_[np.ones(len(data_P)), np.zeros(len(data_U))]
    sample_weight_ = [cost_fn if i else cost_fp for i in y_train]
    rf.fit(X_train, y_train, sample_weight=sample_weight_)
    y_pred, y_prob = rf.predict(data_U), rf.predict_proba(data_U)[:, -1]
    BiasedRF_v1_performance = model_performance(y_pred, pred_U, y_prob)
    print('BiasedRF_v1_performance', BiasedRF_v1_performance)
    df_f_score.append(BiasedRF_v1_performance[0])
    df_gmean.append(BiasedRF_v1_performance[1])
    df_recall.append(BiasedRF_v1_performance[2])
    df_precision.append(BiasedRF_v1_performance[3])
    df_acc.append(BiasedRF_v1_performance[4])
    df_auc.append(BiasedRF_v1_performance[5])
    df_mcc.append(BiasedRF_v1_performance[6])



    ## [3]PU_Incorporation with label frequency
    ### RF


    X_train = np.r_[data_P,data_P, data_U]
    y_train = np.r_[np.ones(len(data_P)),np.zeros(len(data_P)), np.zeros(len(data_U))]
    sample_weight_ = np.r_[[1/label_frequency]*len(data_P),[1-1/label_frequency]*len(data_P),np.ones(len(data_U))]
    rf_class_balanced = RandomForestClassifier(n_estimators=350, max_depth=6, random_state=2018,
                                               class_weight='balanced')
    rf_class_balanced.fit(X_train, y_train , sample_weight = sample_weight_)
    y_pred, y_prob = rf_class_balanced.predict(data_U), rf_class_balanced.predict_proba(data_U)[:, -1]
    Empirical_process_rf_performance = model_performance(y_pred, pred_U, y_prob)
    print('Empirical-process_class_weight_rf_performance :', Empirical_process_rf_performance)
    df_f_score.append(Empirical_process_rf_performance[0])
    df_gmean.append(Empirical_process_rf_performance[1])
    df_recall.append(Empirical_process_rf_performance[2])
    df_precision.append(Empirical_process_rf_performance[3])
    df_acc.append(Empirical_process_rf_performance[4])
    df_auc.append(Empirical_process_rf_performance[5])
    df_mcc.append(Empirical_process_rf_performance[6])



    ### mix_supervised
    x_train = np.r_[data_P, data_U[:len(data_P)]]
    y_train = np.r_[np.ones(len(data_P)), np.zeros(len(data_U[:len(data_P)]))]
    rf.fit(x_train, y_train)
    y_pred, y_prob = rf.predict(data_U), rf.predict_proba(data_U)[:, -1]
    not_mix_trainset_supervised_rf_performance = model_performance(y_pred, pred_U, y_prob)
    print('not_mix_trainset_supervised_rf_performance:', not_mix_trainset_supervised_rf_performance)
    df_f_score.append(not_mix_trainset_supervised_rf_performance[0])
    df_gmean.append(not_mix_trainset_supervised_rf_performance[1])
    df_recall.append(not_mix_trainset_supervised_rf_performance[2])
    df_precision.append(not_mix_trainset_supervised_rf_performance[3])
    df_acc.append(not_mix_trainset_supervised_rf_performance[4])
    df_auc.append(not_mix_trainset_supervised_rf_performance[5])
    df_mcc.append(not_mix_trainset_supervised_rf_performance[6])


    ### unsupervised
    dataset = np.r_[data_P, data_U]
    iforest = IsolationForest(n_estimators=100, contamination=0.01,
                              random_state=2018, n_jobs=-1)
    iforest.fit(dataset)
    # Paper：The higher is the score IS(x) (close to 1), the more likely that x being an anomaly.
    # Scikit-learn API : decision_function(X): The lower, the more abnormal.
    #score∈[0,1]，decision_function返回0.5-score
    # isolation_score = -iforest.decision_function(dataset)
    # isolation_score_scaled = minmax_scale(isolation_score)
    # 最终结果[0,1]    s越大越异常，s≈0.5不能区分，s≈0正常

    isolation_score = -iforest.decision_function(dataset)
    isolation_score_scaled = minmax_scale(isolation_score)
    y_prob = isolation_score_scaled[len(data_P):]
    np.save(r'y_prob.npy',y_prob)
    np.save(r'pred_U.npy',pred_U)
    # f1_value,threshold = find_best_threshold(y_prob, pred_U)
    # print('f1_value:{},threshold:{}'.format(f1_value,threshold))
    performance = find_best_f1(y_prob,pred_U)
    print(performance)
    df_f_score.append(performance[0])
    df_gmean.append(performance[1])
    df_recall.append(performance[2])
    df_precision.append(performance[3])
    df_acc.append(performance[4])
    df_auc.append(performance[5])
    df_mcc.append(performance[6])



    ## [5]pure_supervised

    j_bool = [True if x == 9 else False for x in label_U]
    x_train = np.r_[data_P, data_U[j_bool][:len(data_P) ]]
    y_train = np.r_[np.ones(len(data_P)), np.zeros(len(data_P))]
    rf.fit(x_train, y_train)
    y_pred, y_prob = rf.predict(data_U), rf.predict_proba(data_U)[:, -1]
    pure_trainset_supervised_rf_performance = model_performance(y_pred, pred_U, y_prob)
    print('pure_trainset_supervised_rf_performance:', pure_trainset_supervised_rf_performance)
    df_f_score.append(pure_trainset_supervised_rf_performance[0])
    df_gmean.append(pure_trainset_supervised_rf_performance[1])
    df_recall.append(pure_trainset_supervised_rf_performance[2])
    df_precision.append(pure_trainset_supervised_rf_performance[3])
    df_acc.append(pure_trainset_supervised_rf_performance[4])
    df_auc.append(pure_trainset_supervised_rf_performance[5])
    df_mcc.append(pure_trainset_supervised_rf_performance[6])






    decription = 'The evaluation of the algorithm has been completed.'
    print(decription, 'Running_Time:{:.2f}s\n'.format(time.time()-start))
    df_value.append(df_f_score)
    df_value.append(df_gmean)

    df_value.append(df_recall)
    df_value.append(df_precision)
    df_value.append(df_acc)
    df_value.append(df_auc)
    df_value.append(df_mcc)


    return df_value




if __name__ == '__main__':

    data_path = "SRC_df_label/df_selection.csv"
    label_frequency = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    columns_list =col = ['label_frequency','PUMD','PU_biased','PU_empirical','mix_supervised','unsupervised','pure_supervised']
    # [f_score, gmean, recall,precision, acc,auc,mcc]
    df_evaluate_fscore = pd.DataFrame(columns= columns_list )
    df_evaluate_gmean = pd.DataFrame(
        columns= columns_list )
    df_evaluate_recall = pd.DataFrame(
        columns= columns_list )
    df_evaluate_precision = pd.DataFrame(
        columns= columns_list )
    df_evaluate_acc = pd.DataFrame(
        columns= columns_list )
    df_evaluate_auc = pd.DataFrame(
        columns= columns_list )
    df_evaluate_mcc = pd.DataFrame(
        columns= columns_list )
    for num, c in enumerate(label_frequency):
        print('-------------label_frequency is {}------------------------------:'.format(c))
        df_value = performance_contrast(100, data_path, c)
        # [f_score, gmean, recall,precision, acc,auc,mcc]
        df_evaluate_fscore .loc[num] = df_value[0]
        df_evaluate_gmean.loc[num] = df_value[1]
        df_evaluate_recall.loc[num] = df_value[2]
        df_evaluate_precision.loc[num] = df_value[3]
        df_evaluate_acc.loc[num] = df_value[4]
        df_evaluate_auc.loc[num] = df_value[5]
        df_evaluate_mcc.loc[num] = df_value[6]
    now = time.time()
    df_evaluate_fscore.to_csv(r'Final_result\method_evaluate\\'+str(now)+'-fscore_method_evaluate.csv', index=False)
    df_evaluate_gmean.to_csv(r'Final_result\method_evaluate\\'  + str(now)+'-gmean_method_evaluate.csv',
                              index=False)
    df_evaluate_recall.to_csv(r'Final_result\method_evaluate\\' + str(now)+'-recall_method_evaluate.csv',
                              index=False)
    df_evaluate_precision.to_csv(r'Final_result\method_evaluate\\'  + str(now)+'-precision_method_evaluate.csv',
                              index=False)

    df_evaluate_acc.to_csv(r'Final_result\method_evaluate\\'  + str(now)+'-acc_method_evaluate.csv',
                              index=False)
    df_evaluate_auc.to_csv(r'Final_result\method_evaluate\\'  +str(now)+ '-auc_method_evaluate.csv',
                              index=False)
    df_evaluate_mcc.to_csv(r'Final_result\method_evaluate\\' + str(now)+'-mcc_method_evaluate.csv',
                              index=False)


