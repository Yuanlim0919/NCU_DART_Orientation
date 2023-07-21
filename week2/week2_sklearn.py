# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score,roc_curve, roc_auc_score
from matplotlib import pyplot as plt

class BreastCancerClassifier:
    def __init__(self,classifier,preprocessor,validator):
        self.dataset = load_breast_cancer()
        self.X = self.dataset['data']
        self.Y = self.dataset['target']
        self.classifier = classifier
        self.metrics = {}
        self.preprocessor = preprocessor
        self.validator = validator
    def data_preprocessing(self):
        self.train_X,self.test_X,self.train_Y,self.test_Y = self.preprocessor(self.X,self.Y)
        pass
    def cross_validate(self):
        self.metrics['confusion_matrix'] = []
        self.metrics['accuracy'] = []
        self.metrics['precision'] = []
        self.metrics['recall'] = []
        self.metrics['f1-score'] = []
        self.metrics['AUC-ROC'] = []
        
        for train_set, val_set in self.validator.split(self.train_X,self.train_Y):
            train_X = self.train_X[train_set]
            train_Y = self.train_Y[train_set]
            val_X = self.train_X[val_set]
            self.val_Y = self.train_Y[val_set]
            self.classifier.fit(train_X,train_Y)
            val_pred_Y = self.classifier.predict(val_X)
            self.val_prob_Y = self.classifier.predict_proba(val_X)[::,1]
            # train and validation 要怎麼分割
            self.metrics['confusion_matrix'].append(self.evaluation(confusion_matrix, self.val_Y, val_pred_Y))
            self.metrics['accuracy'].append(self.evaluation(accuracy_score, self.val_Y, val_pred_Y))
            self.metrics['precision'].append(self.evaluation(precision_score, self.val_Y, val_pred_Y))
            self.metrics['recall'].append(self.evaluation(recall_score,self.val_Y,val_pred_Y))
            self.metrics['f1-score'].append(self.evaluation(f1_score,self.val_Y,val_pred_Y))
            self.metrics['AUC-ROC'].append(self.evaluation(roc_auc_score, self.val_Y, val_pred_Y))
            
        return self.metrics
    
    def evaluation(self,metrics,X,Y):
        eva = metrics(X,Y)
        return eva
    def ROC_AUC(self,title):
        fpr,tpr,_ = roc_curve(self.val_Y, self.val_prob_Y)
        
        plt.plot(fpr,tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        #plt.legend(loc=4)
        plt.savefig(title+".png",format="png")
        pass

DTree = DecisionTreeClassifier()
LogReg = LogisticRegression(max_iter=150)
SVM = SVC(probability=True)
k_fold = KFold()

Classifier_DTree = BreastCancerClassifier(classifier=DTree,preprocessor=train_test_split,validator=k_fold)
Classifier_DTree.data_preprocessing()
metrics_DTree = Classifier_DTree.cross_validate()
Classifier_DTree.ROC_AUC("ROC_AUC: Decision Tree Classifier")

Classifier_LogReg = BreastCancerClassifier(LogReg, preprocessor=train_test_split, validator=k_fold)
Classifier_LogReg.data_preprocessing()
metrics_LogReg = Classifier_LogReg.cross_validate()
Classifier_LogReg.ROC_AUC("ROC_AUC: Logistic Regressor")


Classifier_SVM = BreastCancerClassifier(classifier=SVM, preprocessor=train_test_split, validator=k_fold)
Classifier_SVM.data_preprocessing()
metrics_SVM = Classifier_SVM.cross_validate()
Classifier_SVM.ROC_AUC("ROC_AUC of Classifier")

