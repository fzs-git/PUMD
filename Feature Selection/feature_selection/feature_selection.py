from eli5.sklearn import PermutationImportance
import pandas as pd
import numpy as np
import time
from operator import itemgetter
from sklearn.metrics import *
import math
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler,minmax_scale
from tqdm import tqdm

class PIFS():
    def __init__(self,classifier,dataTrain,labelTrain,dataValid,labelValid,feature_list,flag ='normal'):
        self.classifier = classifier
        self.flag = flag
        self.dataTrain = dataTrain
        self.labelTrain = labelTrain
        self.dataValid = dataValid
        self.labelValid = labelValid
        self.PIthr = 7e-6
        self.Corrthr = 0.95
        self.binSizeList =[25,15,10,5]
        self.org_feature_list = feature_list
        self.org_feature_dict = dict(zip(feature_list, range(len(feature_list))))
        self.feature_list = self.org_feature_list
        self.feature_dict = self.org_feature_dict
        self.calcCORRscore()
        self.baseline , self.model= self.getF1Score(self.dataTrain, self.labelTrain, self.dataValid, self.labelValid )
        print("init baseline score:", self.baseline)





    def getF1Score(self,dataTrain,labelTrain,dataValid,labelValid):
        #归一化数值
        scaler = StandardScaler().fit(dataTrain)
        dataTrain = scaler.transform(dataTrain)
        dataValid = scaler.transform(dataValid)

        if self.flag =='normal':
            thisModel = self.classifier.fit(self.dataTrain, self.labelTrain)
            y_pred = thisModel.predict(dataValid)
            f1score = f1_score(labelValid, y_pred)
        else:#PUMD
            anomalies = dataTrain[:sum(labelTrain), :]
            unlabel = dataTrain[sum(labelTrain):, :]
            # STEP1:isolation_score
            isolation_score = self.cal_isolation_score(anomalies, unlabel)
            #STEP2:selction of RN & set weight
            dataTrain,labelTrain,sample_weight = self.determine_trainset(isolation_score,anomalies, unlabel)
            thisModel = self.classifier.fit(dataTrain, labelTrain, sample_weight = sample_weight)
            y_pred = thisModel.predict(dataValid)
            f1score = f1_score(labelValid, y_pred)
        return f1score,thisModel

    def  cal_isolation_score(self,anomalies, unlabel):
        dataset = np.r_[anomalies, unlabel]
        iforest = IsolationForest(n_estimators=100, contamination=0.046,
                                  random_state=2021, n_jobs=-1)
        iforest.fit(dataset)
        isolation_score = -iforest.decision_function(dataset)
        isolation_score_scaled = minmax_scale(isolation_score)
        return isolation_score_scaled

    def determine_trainset(self,isolation_score,anomalies, unlabel):
        min_score, max_score, median_score = [func(isolation_score) for func in (np.min, np.max, np.median)]
        anomalies_score = isolation_score[:len(anomalies)]
        unlabel_scores = isolation_score[len(anomalies):]
        # 1  average isolation score of samples in set P
        #   threshold score of potential malicious domains
        alpha = np.mean(anomalies_score)
        # 2  average isolation score of samples in set D
        #    threshold score of potential benign domains
        isthr = median_score
        # 3  original value of threshold 𝛽
        beta = isthr if isthr < alpha else alpha

        # 4 RN sample set :
        ad_beta = 1.0
        rlb_bool = unlabel_scores <= beta * ad_beta
        rlb_normal = unlabel[rlb_bool]
        rlb_normal_score = unlabel_scores[rlb_bool]

        # 5 sample weight:
        ad_para_org = 19
        self.ad_param_weight = len(anomalies) / np.sum(
            (max_score - rlb_normal_score) / (max_score - min_score)) * ad_para_org
        rlb_normal_weight = (max_score - rlb_normal_score) / (max_score - min_score) * self.ad_param_weight

        weight = np.r_[rlb_normal_weight,[1]*len(anomalies)]
        dataTrain = np.r_[rlb_normal,anomalies]
        labelTrain = np.r_[[0]*len(rlb_normal),[1]*len(anomalies)]

        return dataTrain,labelTrain,weight


    def update(self,deleteList):
        savelistNum = list(set(range(len(self.feature_list))) - set(deleteList))
        TMPfeature_list = [self.feature_list[i] for i in savelistNum]
        self.feature_list = TMPfeature_list
        self.feature_dict = dict(zip(self.feature_list, range(len(self.feature_list))))
        self.CORRList = list(set(self.feature_list) & set(self.CORRList))
        self.dataValid = np.delete(self.dataValid,deleteList,axis = 1)
        self.dataTrain = np.delete(self.dataTrain,deleteList,axis = 1)
        self.baseline , self.model= self.getF1Score(self.dataTrain, self.labelTrain, self.dataValid, self.labelValid )

    def calcCORRscore(self):
        #只计算一次 caculate only once
        pdValid = pd.DataFrame(self.dataValid)
        pdValid = pdValid.apply(lambda x:x.astype(float))
        df_corr = pdValid.corr()
        pdValid.to_csv(r'pdValid.csv',index=False)
        df_corr.to_csv(r'df_corr.csv',index=False)
        CORRcandidate = []
        for i in range(len(df_corr) - 1):
            for j in range(i + 1, len(df_corr) - 1):
                if df_corr.iloc[i, j] > self.Corrthr:
                    CORRcandidate.append(i)
                    CORRcandidate.append(j)
        self.CORRcandidate = list(set(CORRcandidate))
        print("CORRcandidate len:",len(self.CORRcandidate))
        self.CORRList = [self.feature_list[i] for i in self.CORRcandidate]

    def PI_save(self,f_importance, clf_name, path):
        im = pd.DataFrame({'importance': f_importance, 'var': self.feature_list,'indexFL':self.feature_dict.values()})
        im = im.sort_values(by='importance', ascending=False)
        timeStamp = str(time.time())
        im.to_csv(path + clf_name + timeStamp+r'feature_importances.csv')
        PIcandidate = im[im['importance']<self.PIthr].indexFL.to_list()
        self.PIsort = im.indexFL.to_list()
        return PIcandidate
    def calcPIscore(self):
        perm = PermutationImportance(self.model, scoring='f1', cv='prefit', random_state=1).fit(self.dataValid, self.labelValid)
        self.PIcandidate = self.PI_save(perm.feature_importances_, type(self.model).__name__, './PIFS_pi_score_')
        print("PIcandidate len:", len(self.PIcandidate))

    def candidateListFunc(self):
        keys = self.CORRList
        if len(keys)>0:
            self.CORRcandidate = itemgetter(*keys)(self.feature_dict)
            self.candidateList = list(set(self.CORRcandidate)|set(self.PIcandidate))
        else:
            self.candidateList = self.PIcandidate

    def PI_FS_EVAL(self):
        self.calcPIscore()
        self.candidateListFunc()


    def NatSort(self):
        self.candidateList.sort(reverse = False)
        flnat =  self.candidateList
        return flnat
    def PISort(self):
        flpi = [f for f in self.PIsort if f in self.candidateList]
        return flpi


    def BIN_FS(self):
        flnat = self.NatSort()
        flpi = self.PISort()
        score_list = []
        fl_list = []
        print("bin split:")
        # self.df_log = pd.DataFrame(["round","binSize", "type", "score", "del_fl"])

        for bin_size in tqdm(self.binSizeList):
            Nat_score,Nat_fl = self.BIN_SBS(bin_size,flnat)
            self.df_log.loc[len(self.df_log)] = [self.round, bin_size, "nat", Nat_score, ",".join('%s' %id for id in Nat_fl)]
            PI_score,PI_fl = self.BIN_SBS(bin_size,flpi)
            self.df_log.loc[len(self.df_log)] = [self.round, bin_size, "pi", PI_score, ",".join('%s' %id for id in PI_fl)]
            score_list.append(Nat_score)
            score_list.append(PI_score)
            fl_list.append(Nat_fl)
            fl_list.append(PI_fl)
        best_score = max(score_list)
        best_fl = fl_list[score_list.index(best_score)]
        return best_score,best_fl

    def BIN_SBS(self,bin_size,candidate_set):
        del_list = self.equalFreqBin(bin_size,candidate_set)
        score_list = []
        print("bining selection:")
        for del_fs in tqdm(del_list):
            thisDataTrain = np.delete(self.dataTrain, del_fs, axis=1)
            thisDataValid = np.delete(self.dataValid, del_fs, axis=1)
            score,model = self.getF1Score(thisDataTrain, self.labelTrain, thisDataValid, self.labelValid )
            score_list.append(score)
        best_score = max(score_list)
        best_fl = del_list[score_list.index(best_score)]
        return best_score,best_fl

    def equalFreqBin(self,size,setC):
        #两种模式，这是box模式
        #还有extend模式
        binList = []
        for i in range(int(math.ceil(len(setC)/size))):
            if (i+1) * size <= len(setC):
                binList.append(setC[i*size: (i+1) * size])
            else:
                binList.append(setC[i*size: len(setC)])
        return binList

    def mainFS(self):
        self.PI_FS_EVAL()
        self.round = 0
        self.df_log = pd.DataFrame(columns=["round", "binSize", "type", "score", "del_fl"])
        best_score,best_fl = self.BIN_FS()
        while best_score > self.baseline and len(self.feature_list)>0:
            self.round = self.round + 1
            print("Feature Selection Round:",self.round)
            self.update(best_fl)
            print("now_f1score:",self.baseline)
            self.PI_FS_EVAL()
            if len(self.candidateList) == 0:
                feature_index = itemgetter(*self.feature_list)(self.org_feature_dict)
                return self.baseline,self.feature_list,feature_index
            best_score, best_fl = self.BIN_FS()



        feature_index = itemgetter(*self.feature_list)(self.org_feature_dict)
        end = time.time()
        self.df_log.to_csv(r'feature_selection_'+str(end)+r'saveLog.csv',index=False)
        return self.baseline, self.feature_list, feature_index




























