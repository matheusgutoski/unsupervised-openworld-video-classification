import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

def closed_f1_score(true, predicted, unknown_label = 0):
        f1 = f1_score(true, predicted, average = 'macro', labels = np.unique([x for x in true if x != unknown_label]))
        return f1

def youdens_index(true, predicted, unknown_label = 0):
        def binary_performance_measure(y_true, y_pred):
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                # True Positives
                for i in range(y_true.shape[0]):
                        if (y_true[i] == y_pred[i] == 1):
                                TP += 1
                # False Positives
                for i in range(y_true.shape[0]):
                        if (y_pred[i] == 1 and y_true[i] != y_pred[i]):
                                FP += 1
                # True Negatives
                for i in range(y_true.shape[0]):
                        if (y_true[i] == y_pred[i] == 0):
                                TN += 1
                # False Negatives
                for i in range(y_true.shape[0]):
                        if (y_pred[i] == 0 and y_true[i] != y_pred[i]):
                                FN += 1
                return TP, FP, TN, FN

        # Sensitivity, Recall or True Positive Rate
        def Sensitivity(y_true, y_pred):
                TP, FP, TN, FN = binary_performance_measure(y_true, y_pred)
                return float(TP) / float(TP+FN)

        # Specificity or True Negative Rate
        def Specificity(y_true, y_pred):
                TP, FP, TN, FN = binary_performance_measure(y_true, y_pred)
                #print TP, FP, TN, FN
                return float(TN) / float(TN+FP)


        true = [unknown_label if x == unknown_label else 1 for x in true] 
        predicted = [unknown_label if x == unknown_label else 1 for x in predicted]

        try:
                J = Sensitivity(true, predicted) + Specificity(true, predicted) - 1
        except:
                J = 'Open Youdens index seems to be undefined. Is the number of classes in test time greater than the number of classes in training time?'
        return J


        

def classif_report(y_test, pred):
        return classification_report(y_test, pred), confusion_matrix(y_test, pred)


'''
true = [0,0,0,3,1,2,3]
predicted = [1,1,1,0,0,0,0]
#predicted = [0,0,0,5,5,5,5]

print youdens_index(true, predicted)
print closed_f1_score(true, predicted, unknown_label = 0)
'''
