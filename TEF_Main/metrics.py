from sklearn.metrics import accuracy_score, recall_score, precision_score, cohen_kappa_score, f1_score


def get_results(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)

    k = int(2)
    results = [round(acc*100, k), round(recall*100, k), round(precision*100, k), round(f1*100, k), round(kappa*100, k)]

    return results
