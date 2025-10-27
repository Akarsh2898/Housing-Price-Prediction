import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the data
print("Loading heart.csv...")
df = pd.read_csv('heart.csv')

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Test different tree depths
depths = range(1, 21)
train_scores = []
test_scores = []

print("\nTesting different tree depths...")
print("-" * 60)
print(f"{'Depth':<8} {'Train Acc':<12} {'Test Acc':<12} {'Difference':<12}")
print("-" * 60)

for depth in depths:
    # Train classifier
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42, criterion='entropy')
    dt.fit(X_train, y_train)
    
    # Training accuracy
    train_pred = dt.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_scores.append(train_acc)
    
    # Testing accuracy
    test_pred = dt.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_scores.append(test_acc)
    
    diff = train_acc - test_acc
    print(f"{depth:<8} {train_acc:.4f}       {test_acc:.4f}       {diff:.4f}")

# Find the best depth (where test accuracy is highest)
best_depth_idx = np.argmax(test_scores)
best_depth = depths[best_depth_idx]

print("\n" + "=" * 60)
print(f"Best depth: {best_depth}")
print(f"Best test accuracy: {test_scores[best_depth_idx]:.4f}")
print(f"Train accuracy at best depth: {train_scores[best_depth_idx]:.4f}")
print(f"Overfitting (difference): {train_scores[best_depth_idx] - test_scores[best_depth_idx]:.4f}")

# Visualize the overfitting analysis
plt.figure(figsize=(12, 6))
plt.plot(depths, train_scores, 'o-', label='Training Accuracy', linewidth=2, markersize=8)
plt.plot(depths, test_scores, 's-', label='Testing Accuracy', linewidth=2, markersize=8)
plt.axvline(x=best_depth, color='red', linestyle='--', label=f'Best Depth ({best_depth})')
plt.xlabel('Tree Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Overfitting Analysis: Training vs Testing Accuracy', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(1, max(depths))
plt.ylim(min(min(train_scores), min(test_scores)) - 0.05, 1.05)
plt.xticks(depths)
plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
print("\nOverfitting analysis plot saved as 'overfitting_analysis.png'")
plt.show()

# Plot the difference (overfitting gap)
plt.figure(figsize=(12, 6))
differences = [train_scores[i] - test_scores[i] for i in range(len(depths))]
plt.plot(depths, differences, 'ro-', linewidth=2, markersize=8)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axvline(x=best_depth, color='red', linestyle='--', label=f'Best Depth ({best_depth})')
plt.xlabel('Tree Depth', fontsize=12)
plt.ylabel('Accuracy Difference (Train - Test)', fontsize=12)
plt.title('Overfitting Gap: Training vs Testing Accuracy Difference', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(1, max(depths))
plt.xticks(depths)
plt.tight_layout()
plt.savefig('overfitting_gap.png', dpi=300, bbox_inches='tight')
print("Overfitting gap plot saved as 'overfitting_gap.png'")
plt.show()

# Train final model with optimal depth
print("\n" + "=" * 60)
print("Training final model with optimal depth...")
final_dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42, criterion='entropy')
final_dt.fit(X_train, y_train)

train_final = accuracy_score(y_train, final_dt.predict(X_train))
test_final = accuracy_score(y_test, final_dt.predict(X_test))

print(f"Final model - Training accuracy: {train_final:.4f}")
print(f"Final model - Testing accuracy: {test_final:.4f}")
print(f"Overfitting gap: {train_final - test_final:.4f}")

# Visualize the final tree
plt.figure(figsize=(20, 10))
from sklearn.tree import plot_tree
plot_tree(final_dt, 
          feature_names=X.columns,
          class_names=['No Heart Disease', 'Heart Disease'],
          filled=True,
          rounded=True,
          fontsize=8)
plt.title(f'Decision Tree with Optimal Depth ({best_depth})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_optimal_depth.png', dpi=300, bbox_inches='tight')
print(f"Final tree visualization saved as 'decision_tree_optimal_depth.png'")
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_dt.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("Feature Importance (Optimal Depth):")
print(feature_importance.to_string(index=False))

# Summary
print("\n" + "=" * 60)
print("OVERFITTING ANALYSIS SUMMARY")
print("=" * 60)
print(f"Optimal tree depth: {best_depth}")
print(f"Test accuracy: {test_final:.4f}")
print(f"Train accuracy: {train_final:.4f}")
print(f"Overfitting gap: {train_final - test_final:.4f}")
print(f"Percentage overfitting: {(train_final - test_final) / train_final * 100:.2f}%")

# Create detailed comparison table
comparison = pd.DataFrame({
    'Depth': list(depths),
    'Train_Accuracy': train_scores,
    'Test_Accuracy': test_scores,
    'Difference': differences
})

print("\n" + "=" * 60)
print("Detailed Comparison (Top 10 by Test Accuracy):")
print("=" * 60)
comparison_sorted = comparison.sort_values('Test_Accuracy', ascending=False).head(10)
print(comparison_sorted.to_string(index=False))

# Save comparison to CSV
comparison.to_csv('depth_comparison.csv', index=False)
print("\nFull comparison saved to 'depth_comparison.csv'")

print("\nAnalysis complete!")

