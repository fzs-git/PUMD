# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, minmax_scale
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV



class PUMD:
    """Implementation of ADOA (Anomaly Detection with Partially Observed Anomalies)"""
    def __init__(self, anomalies, unlabel, classifer,
                 contamination=0.01, theta=0.99, alpha='auto', beta='auto', return_proba=False,
                 random_state=2018,pred_U = [],ad_para = 1.0 ,feature_list = [],ad_beta = 1.0,
                 mode='fs',generalize_mode =0,gdata = np.empty([1,1]),
                 label_frequency = 1):

        dataset_scaled = StandardScaler().fit_transform(np.r_[anomalies, unlabel])
        self.gdata = StandardScaler().fit_transform(gdata)
        self.anomalies = dataset_scaled[:len(anomalies), :]
        self.unlabel = dataset_scaled[len(anomalies):, :]
        self.contamination = contamination
        self.classifer = classifer
        self.theta = theta 
        self.alpha = alpha 
        self.beta = beta
        self.ad_beta = ad_beta
        self.return_proba = return_proba 
        self.random_state = random_state
        self.pred_U = pred_U
        self.ad_para = ad_para
        self.feature_list = feature_list
        self.scores = 0
        self.mode = mode
        self.generalize_mode = generalize_mode
        self.ad_param_weight = 1.0
        self.rlb_normal_weight = 0
        self.anomalies_weight = 0
        self.rlb_bool = 0
        self.label_frequency = label_frequency

    def cal_isolation_score(self):
        dataset = np.r_[self.anomalies, self.unlabel]
        iforest = IsolationForest(n_estimators=100, contamination=self.contamination, 
                                  random_state=self.random_state, n_jobs=-1)
        iforest.fit(dataset)  
        # Paper：The higher is the score IS(x) (close to 1), the more likely that x being an anomaly.
        # Scikit-learn API : decision_function(X): The lower, the more abnormal.
        isolation_score = -iforest.decision_function(dataset)  
        isolation_score_scaled = minmax_scale(isolation_score)

        return isolation_score_scaled
    
    def determine_trainset(self):
        weighted_score = self.cal_isolation_score()

        min_score, max_score, median_score = [func(weighted_score) for func in (np.min, np.max, np.median)]

        anomalies_score = weighted_score[:len(self.anomalies)]

        unlabel_scores = weighted_score[len(self.anomalies):]
        self.scores = weighted_score
        # determine the value of alpha、beta
        self.alpha = np.mean(anomalies_score) if self.alpha == 'auto' else self.alpha
        percent = 45
        self.beta = median_score if median_score < self.alpha else np.percentile(weighted_score, percent)
        while self.beta >= self.alpha:
            percent -= 5
            self.beta = np.percentile(weighted_score, percent)
        assert self.beta < self.alpha, 'beta should be smaller than alpha.'

        rlb_bool= unlabel_scores<= self.beta*self.ad_beta

        self.rlb_bool = rlb_bool
        rlb_normal= self.unlabel[rlb_bool]
        rlb_normal_score= unlabel_scores[rlb_bool]
        self.ad_param_weight = len(self.anomalies)/np.sum((max_score-rlb_normal_score) / (max_score-min_score))*self.ad_para
        rlb_normal_weight = (max_score-rlb_normal_score) / (max_score-min_score) * self.ad_param_weight


        anomalies_weight = anomalies_label = np.ones(len(self.anomalies))
        X_train = np.r_[self.anomalies, rlb_normal]
        weights = np.r_[anomalies_weight, rlb_normal_weight]
        # weights = np.r_[anomalies_weight,np.ones(len(rlb_normal_weight))]
        self.rlb_normal_weight = np.sum(rlb_normal_weight)
        self.anomalies_weight = np.sum(anomalies_weight)

        y_train = np.r_[anomalies_label,  np.zeros(len(rlb_normal))].astype(int)
        return X_train, y_train, weights


    
    def predict(self):
        X_train, y_train, weights = self.determine_trainset()
        clf = self.classifer
        clf.fit(X_train, y_train, sample_weight=weights)
        if self.generalize_mode == 1 :
            print('generalize_mode')
            predict_data = self.gdata
        else:
            predict_data = self.unlabel
        y_pred = clf.predict(predict_data)
        if self.return_proba:
            y_prob = clf.predict_proba(predict_data)[:, 1]
            return y_pred, y_prob
        else:
            return y_pred
        
    def __repr__(self):

        
        y_train = self.determine_trainset()[1]
        rll_num = np.sum(y_train==0)
        ptt_num = sum(y_train)-len(self.anomalies)
      
        info_2 = "2) Reliable Normals's number = {:}, accounts for {:.2%} within the Unlabel dataset.\n".\
        format(rll_num, rll_num/len(self.unlabel))
        unlabel_scores = self.scores[len(self.anomalies):]
        min_score, max_score = [func(self.scores) for func in (np.min, np.max)]
        beta=self.beta
        pred_U=self.pred_U
        # rlb_list = [0 if score <= self.beta*self.ad_para else 1 for score in unlabel_scores]
        rlb_list = [0 if bool else 1 for bool in self.rlb_bool]

        cnt_a = 0
        cnt_b = []
        for i, j in zip(rlb_list, pred_U):
            if i == 0:
                if j == 1:
                    cnt_b.append(cnt_a)
            cnt_a = cnt_a + 1
        print('----------------------evaluate_trainset--------------------------')
        print('|   beta_org   |', self.beta)
        print('|   beta_RN    |', self.beta*self.ad_beta)
        print('|   train_RN   |', rll_num)
        print('|  num_mis_RN  |', len(cnt_b))
        print('|    mis_RN    |', cnt_b)
        print('|    score     |', [unlabel_scores[i] for i in cnt_b])
        print('|    weight    |', [(max_score - unlabel_scores[i])/(max_score - min_score) * self.ad_param_weight for i in cnt_b])
        print('| P_weight vs N_weight     |', self.anomalies_weight,'vs', self.rlb_normal_weight )
        print('----------------------evaluate_trainset--------------------------\n')

