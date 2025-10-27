import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
print("Loading heart.csv...")
df = pd.read_csv('heart.csv')

# Check the data
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nTarget distribution:")
print(df['target'].value_counts())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Get feature names
feature_names = X.columns.tolist()
print(f"\nFeature names: {feature_names}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
print("\nTraining Decision Tree Classifier...")
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42, criterion='entropy')
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, 
          feature_names=feature_names,
          class_names=['No Heart Disease', 'Heart Disease'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Classifier - Heart Disease Prediction', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
print("\nDecision tree visualization saved as 'decision_tree_visualization.png'")
plt.show()

# Feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved as 'feature_importance.png'")
plt.show()

print("\nTraining complete!")


