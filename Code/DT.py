import sys
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_predict

nb_running = 1


def define_model(criterion):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion=criterion, max_depth=100, random_state=42)
    return model


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    from sklearn.metrics import roc_auc_score
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
    return roc_auc_dict


def evaluation_model(y, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    import numpy as np
    accuracy = accuracy_score(y_pred, y) * 100
    precision = precision_score(y_pred, y, average='weighted', zero_division=1) * 100
    recall = recall_score(y_pred, y, average='weighted', zero_division=1) * 100
    f1 = f1_score(y_pred, y, average='weighted', zero_division=1) * 100
    try:
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y_pred, y, multi_class="raise") * 100
        else:
            auc = roc_auc_score_multiclass(y, y_pred)
            auc = sum(auc.values()) / len(auc) * 100
    except:
        auc = 0
    return round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2), round(auc, 2)


def fit_model(X, y, criterion):
    list_accuracy = []
    list_precision = []
    list_recall = []
    list_f1 = []
    list_auc = []

    cv = evaluation_protocol(X)

    for i in range(0, nb_running):
        print("--------------", i, "-----------------")
        model = define_model(criterion)
        model.fit(X, y)
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=8)
        accuracy, precision, recall, f1, auc = evaluation_model(y, y_pred)
        list_accuracy.append(accuracy)
        list_precision.append(precision)
        list_recall.append(recall)
        list_f1.append(f1)
        list_auc.append(auc)

    mean_accuracy = round(np.mean(list_accuracy), 2)
    mean_precison = round(np.mean(list_precision), 2)
    mean_recall = round(np.mean(list_recall), 2)
    mean_f1 = round(np.mean(list_f1), 2)
    mean_auc = round(np.mean(list_auc), 2)

    std_accuracy = np.std(list_accuracy)
    std_precison = np.std(list_precision)
    std_recall = np.std(list_recall)
    std_f1 = np.std(list_f1)
    std_auc = np.std(list_auc)

    return mean_accuracy, mean_precison, mean_recall, mean_f1, mean_auc, std_accuracy, std_precison, std_recall, std_f1, std_auc


# Tien xu ly du lieu
def preprocessing_load_data(file):
    import pandas as pd
    df_train = pd.read_csv(file, sep=",", low_memory=False)
    print(df_train.head(3))
    df_train = df_train.replace(['?'], -9999)
    X = df_train.drop('class', axis=1).values
    y = df_train['class']
    return X, y


def preprocessing_label_encoder(y):
    from sklearn.preprocessing import LabelEncoder
    m_LabelEncoder = LabelEncoder()
    m_LabelEncoder.fit(y)
    y = m_LabelEncoder.transform(y)
    return y


def preprocessing_standard_scaler(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def evaluation_protocol(X):
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    if X.shape[0] >= 300:
        cv = StratifiedKFold(n_splits=10, shuffle=True)
    else:
        cv = LeaveOneOut()
    return cv


def main():
    file_data = sys.argv[1]
    arr = file_data.split("/")
    file_name = arr.pop()
    X, y = preprocessing_load_data(file_data)
    X = preprocessing_standard_scaler(X)
    y = preprocessing_label_encoder(y)

    list_cri = ['gini', 'entropy']
    for cri in list_cri:
        mean_accuracy, mean_precison, mean_recall, mean_f1, mean_auc, std_accuracy, std_precison, std_recall, std_f1, std_auc = fit_model(
            X, y, cri)
        str_mean = str(mean_accuracy) + "_" + str(mean_precison) + "_" + str(mean_recall) + "_" + str(
            mean_f1) + "_" + str(mean_auc)
        str_std = str(std_accuracy) + "_" + str(std_precison) + "_" + str(std_recall) + "_" + str(std_f1) + "_" + str(
            std_auc)
        str_pars = "_" + cri + "_"
        # Ghi Log
        file_result = open(file_name + "_DT_" + str_mean + str(str_pars) + "_.txt", 'w')
        file_result.write("\n-----------------------------------------------")
        file_result.write('\nFile               ' + str(file_name))
        file_result.write("\n-----------------------------------------------")
        file_result.write('\nMEAN:           |' + str(str_mean))
        file_result.write("\n-----------------------------------------------")
        file_result.write('\nSTD:           |' + str(str_std))
        file_result.write("\n-----------------------------------------------")
        file_result.close()


main()
