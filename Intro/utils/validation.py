def binary_classification_report(y_true, y_pred):
    # count true positives, false positives, true negatives, and false negatives
    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred):
        if pred == True:
            if true == True:
                tp += 1
            else:
                fp += 1
        else:
            if true == False:
                tn += 1
            else:
                fn += 1

    # calculate precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # calculate f1 score
    fscore = 2 * precision * recall / (precision + recall)

    # calculate accuracy
    accuracy = (tp + tn) / len(y_true)

    # number of positive labels in y_true
    support = sum(y_true)
    return {
        "precision": precision,
        "recall": recall,
        "f1-score": fscore,
        "support": support,
        "accuracy": accuracy,
    }