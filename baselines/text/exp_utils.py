import csv
import glob
import os
import json


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as compute_confusion_matrix,
    precision_recall_fscore_support, f1_score,
)

import numpy as np
import pandas as pd




# Convert a dictionary to JSON-able object, converting all numpy arrays to python
# lists
def convert_to_json(obj):
    converted_obj = {}

    for key, value in obj.items():
        if isinstance(value, dict):
            print(key, type(value))
            converted_obj[key] = convert_to_json(value)
        # elif isinstance(value, list):
        #     # print(key, type(value))
        #     converted_obj[key] = value
        else:
            # print(key, type(value))
            # print(value)
            converted_obj[key] = np.array(value).tolist()
            # getattr(value, "tolist", lambda: value)()
            # print(key, type(converted_obj[key]))

    return converted_obj

def read_labels(tsv_path, label_column):
    data = []
    with open(tsv_path) as tsvfile:
        input_file = csv.reader(tsvfile, delimiter="\t", quotechar="")
        for row_idx, row in enumerate(input_file):
            if row_idx == 0:
                # Ignore header row
                continue
            if(len(row)==0):
                continue
            label=row[label_column].strip()
            if (label == "nan" or label == "NA" or label == ""):
                continue

            data.append(label)
    return data

# Utility function to compute precision, recall and F1 from a confusion matrix
def cm_to_precision_recall_f1(cm, all_classes):
    assert len(all_classes) == cm.shape[0]
    num_classes = len(all_classes)
    constructed_y_true = []
    constructed_y_pred = []
    for class_idx in range(num_classes):
        constructed_y_true.extend([all_classes[class_idx]] * np.sum(cm[class_idx, :]))
        for pred_class_idx in range(num_classes):
            constructed_y_pred.extend(
                [all_classes[pred_class_idx]] * cm[class_idx, pred_class_idx]
            )

    constructed_y_true = np.array(constructed_y_true)
    constructed_y_pred = np.array(constructed_y_pred)

    return precision_recall_fscore_support(
        constructed_y_true, constructed_y_pred, labels=all_classes, average="weighted"
    )[:-1]


# Function to compute aggregated scores from per-fold evaluations
def compute_aggregate_scores(all_labels, all_predictions, all_classes):
    accuracy = accuracy_score(all_labels, all_predictions)
    confusion_matrix = compute_confusion_matrix(all_labels, all_predictions, labels=all_classes)
    prf_per_class = precision_recall_fscore_support(
            all_labels, all_predictions, labels=all_classes, average=None
        )[:-1]
    prf_micro = precision_recall_fscore_support(
            all_labels, all_predictions, labels=all_classes, average='micro'
        )[:-1]
    prf_macro = precision_recall_fscore_support(
            all_labels, all_predictions, labels=all_classes, average='macro'
        )[:-1]
    prf_weighted = precision_recall_fscore_support(
            all_labels, all_predictions, labels=all_classes, average='weighted'
        )[:-1]
    aggregated_metrics = {
        "accuracy": accuracy,
        "prf_per_class": prf_per_class,
        "prf_per_class_labels": all_classes,
        "prf_micro": prf_micro,
        "prf_macro": prf_macro,
        "prf_weighted": prf_weighted,
        "confusion_matrix": confusion_matrix,
    }

    return aggregated_metrics

# Function to compute aggregated scores from per-fold evaluations
def compute_aggregate_scores_multilabel(all_labels, all_predictions, all_classes):
    all_labels=np.array(all_labels)
    all_predictions=np.array([l.tolist() for l in all_predictions])
    accuracy = accuracy_score(all_labels, all_predictions)
    prf_micro = f1_score(all_labels, all_predictions, average='micro')
    prf_macro = f1_score(all_labels, all_predictions, average='macro')
    aggregated_metrics = {
        "accuracy": accuracy,
        # "prf_per_class": prf_per_class,
        "prf_per_class_labels": all_classes,
        "prf_micro": prf_micro,
        "prf_macro": prf_macro,
        # "prf_weighted": prf_weighted,
        # "confusion_matrix": confusion_matrix,
    }

    return aggregated_metrics

