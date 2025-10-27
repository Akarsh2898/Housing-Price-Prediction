import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

# Load the data
print("=" * 70)
print("HEART DISEASE PREDICTION - FINAL MODEL (Depth 10)")
print("=" * 70)

df = pd.read_csv('heart.csv')
print(f"\nDataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Features: {X.columns.tolist()}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train the final Decision Tree with optimal depth
print("\n" + "=" * 70)
print("Training Decision Tree Classifier with DEPTH = 10")
print("=" * 70)

dt_model = DecisionTreeClassifier(
    max_depth=10,
    criterion='entropy',
    random_state=42,
    min_samples_split=2,
    min_samples_leaf=1
)

dt_model.fit(X_train, y_train)

# Predictions
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"\nTraining Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Overfitting Gap:   {train_acc - test_acc:.4f} ({(train_acc - test_acc)/train_acc*100:.2f}%)")

# Detailed classification report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_test_pred, 
                          target_names=['No Heart Disease', 'Heart Disease']))

# Confusion matrix
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
cm = confusion_matrix(y_test, y_test_pred)
print("\nActual vs Predicted:")
print(f"                 Predicted: 0    Predicted: 1")
print(f"Actual: 0          {cm[0,0]:4d}       {cm[0,1]:4d}")
print(f"Actual: 1          {cm[1,0]:4d}       {cm[1,1]:4d}")

# Feature importance
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

# Visualize the decision tree
print("\nGenerating decision tree visualization...")
plt.figure(figsize=(24, 12))
plot_tree(dt_model, 
          feature_names=X.columns,
          class_names=['No Heart Disease', 'Heart Disease'],
          filled=True,
          rounded=True,
          fontsize=9,
          max_depth=5)  # Limit visualization to 5 levels for readability
plt.title('Decision Tree Classifier - Heart Disease Prediction (Depth 10)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('decision_tree_depth10.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Decision tree saved as 'decision_tree_depth10.png'")
plt.close()

# Feature importance bar chart
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance - Final Model (Depth 10)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, feature_importance['Importance'])):
    plt.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{value:.3f}', va='center', fontsize=9)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_depth10.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Feature importance saved as 'feature_importance_depth10.png'")
plt.close()

# Save the model
model_filename = 'heart_disease_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(dt_model, f)
print(f"[SUCCESS] Model saved as '{model_filename}'")

# Model summary
print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)
print(f"Model Type: Decision Tree Classifier")
print(f"Depth: 10")
print(f"Criterion: Entropy")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy:  {test_acc:.4f}")
print(f"Overfitting: {(train_acc - test_acc)/train_acc*100:.2f}%")
print(f"Status: [READY] Production Ready")

# Create predictions example
print("\n" + "=" * 70)
print("EXAMPLE PREDICTION")
print("=" * 70)
example = X_test.iloc[0:1]
prediction = dt_model.predict(example)[0]
probability = dt_model.predict_proba(example)[0]

print("\nSample patient features:")
print(example.iloc[0])
print(f"\nPrediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
print(f"Confidence: {probability[prediction]*100:.2f}%")
print(f"Actual: {'Heart Disease' if y_test.iloc[0] == 1 else 'No Heart Disease'}")

print("\n" + "=" * 70)
print("[SUCCESS] ALL DONE!")
print("=" * 70)

