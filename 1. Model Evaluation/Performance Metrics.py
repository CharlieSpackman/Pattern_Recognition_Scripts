# Performance Metrics

import numpy as np
from sklearn import metrics

# Amend outcomes
predicted_outcomes = np.array([1,1,1,1,1,0,0,0])

actual_outcomes = np.array([0,0,0,1,0,1,0,0])


# Calculate performance metrics
accuracy = metrics.accuracy_score(actual_outcomes, predicted_outcomes)
confusion_matrix = metrics.confusion_matrix(actual_outcomes, predicted_outcomes).ravel()
precision = metrics.precision_score(actual_outcomes, predicted_outcomes)
recall = metrics.recall_score(actual_outcomes, predicted_outcomes)
f1 = metrics.f1_score(actual_outcomes, predicted_outcomes)

# Print performance metrics
print("\n#### PRINTING METRICS ####\n")

print(f"Error rate: {round(1 - accuracy, 4)}\n")
print(f"Accuracy score: {round(accuracy, 4)}\n")
print(f"Confusion matrix:")
print(f"tn:{confusion_matrix[0]}")
print(f"fp:{confusion_matrix[1]}")
print(f"fn:{confusion_matrix[2]}")
print(f"tp:{confusion_matrix[3]}")

print(f"\nPrecision score: {round(precision, 4)}")
print(f"\nRecall score: {round(recall, 4)}")
print(f"\nF1 score: {round(f1,4)}")