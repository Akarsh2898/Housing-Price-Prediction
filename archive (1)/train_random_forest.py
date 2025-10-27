import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("=" * 70)
print("RANDOM FOREST vs DECISION TREE COMPARISON")
print("=" * 70)

df = pd.read_csv('heart.csv')
print(f"\nDataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ============================================================================
# TRAIN DECISION TREE (Depth 10)
# ============================================================================
print("\n" + "=" * 70)
print("Training Decision Tree Classifier (Depth 10)...")
print("=" * 70)

dt_model = DecisionTreeClassifier(
    max_depth=10,
    criterion='entropy',
    random_state=42
)

dt_model.fit(X_train, y_train)
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)

dt_train_acc = accuracy_score(y_train, dt_train_pred)
dt_test_acc = accuracy_score(y_test, dt_test_pred)

print(f"Decision Tree - Training Accuracy: {dt_train_acc:.4f} ({dt_train_acc*100:.2f}%)")
print(f"Decision Tree - Testing Accuracy:  {dt_test_acc:.4f} ({dt_test_acc*100:.2f}%)")
print(f"Decision Tree - Overfitting Gap:  {dt_train_acc - dt_test_acc:.4f}")

# ============================================================================
# TRAIN RANDOM FOREST
# ============================================================================
print("\n" + "=" * 70)
print("Training Random Forest Classifier...")
print("=" * 70)

# Try different n_estimators
n_estimators_list = [50, 100, 150, 200]
rf_scores = []

for n_est in n_estimators_list:
    rf_model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=10,
        criterion='entropy',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    rf_test_pred = rf_model.predict(X_test)
    rf_test_acc = accuracy_score(y_test, rf_test_pred)
    rf_scores.append(rf_test_acc)
    
    print(f"Random Forest (n_estimators={n_est:3d}) - Test Accuracy: {rf_test_acc:.4f} ({rf_test_acc*100:.2f}%)")

# Find best n_estimators
best_n_estimators_idx = np.argmax(rf_scores)
best_n_estimators = n_estimators_list[best_n_estimators_idx]

print(f"\nOptimal n_estimators: {best_n_estimators}")

# Train final Random Forest model
print("\n" + "=" * 70)
print(f"Training Final Random Forest (n_estimators={best_n_estimators})...")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=10,
    criterion='entropy',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

print(f"Random Forest - Training Accuracy: {rf_train_acc:.4f} ({rf_train_acc*100:.2f}%)")
print(f"Random Forest - Testing Accuracy:  {rf_test_acc:.4f} ({rf_test_acc*100:.2f}%)")
print(f"Random Forest - Overfitting Gap:   {rf_train_acc - rf_test_acc:.4f}")

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)

comparison_data = {
    'Model': ['Decision Tree', 'Random Forest'],
    'Training Accuracy': [dt_train_acc, rf_train_acc],
    'Testing Accuracy': [dt_test_acc, rf_test_acc],
    'Overfitting Gap': [dt_train_acc - dt_test_acc, rf_train_acc - rf_test_acc]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Training Accuracy'] = comparison_df['Training Accuracy'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
comparison_df['Testing Accuracy'] = comparison_df['Testing Accuracy'].apply(lambda x: f"{x:.4f} ({x*100:.2f}%)")
comparison_df['Overfitting Gap'] = comparison_df['Overfitting Gap'].apply(lambda x: f"{x:.4f} ({x/100:.2%})")

print("\n" + comparison_df.to_string(index=False))

# Calculate improvement
improvement = rf_test_acc - dt_test_acc
improvement_pct = (improvement / dt_test_acc) * 100

print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"\nAccuracy Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)")
print(f"Random Forest {'OUTPERFORMS' if improvement > 0 else 'UNDERPERFORMS'} Decision Tree")

# ============================================================================
# DETAILED CLASSIFICATION REPORTS
# ============================================================================
print("\n" + "=" * 70)
print("DECISION TREE - CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, dt_test_pred, 
                          target_names=['No Heart Disease', 'Heart Disease']))

print("\n" + "=" * 70)
print("RANDOM FOREST - CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, rf_test_pred, 
                          target_names=['No Heart Disease', 'Heart Disease']))

# ============================================================================
# CONFUSION MATRICES
# ============================================================================
print("\n" + "=" * 70)
print("CONFUSION MATRICES COMPARISON")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision Tree CM
cm_dt = confusion_matrix(y_test, dt_test_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[0].set_title('Decision Tree', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Random Forest CM
cm_rf = confusion_matrix(y_test, rf_test_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[1].set_title('Random Forest', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
print("\n[SUCCESS] Confusion matrices saved as 'confusion_matrices_comparison.png'")
plt.close()

# Print confusion matrices
print("\nDecision Tree Confusion Matrix:")
print("                 Predicted: 0    Predicted: 1")
print(f"Actual: 0          {cm_dt[0,0]:4d}       {cm_dt[0,1]:4d}")
print(f"Actual: 1          {cm_dt[1,0]:4d}       {cm_dt[1,1]:4d}")

print("\nRandom Forest Confusion Matrix:")
print("                 Predicted: 0    Predicted: 1")
print(f"Actual: 0          {cm_rf[0,0]:4d}       {cm_rf[0,1]:4d}")
print(f"Actual: 1          {cm_rf[1,0]:4d}       {cm_rf[1,1]:4d}")

# ============================================================================
# FEATURE IMPORTANCE COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE COMPARISON")
print("=" * 70)

# Decision Tree feature importance
dt_importance = pd.DataFrame({
    'Feature': X.columns,
    'Decision_Tree': dt_model.feature_importances_
})

# Random Forest feature importance
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Random_Forest': rf_model.feature_importances_
})

importance_comparison = pd.merge(dt_importance, rf_importance, on='Feature')
importance_comparison['Difference'] = importance_comparison['Random_Forest'] - importance_comparison['Decision_Tree']
importance_comparison = importance_comparison.sort_values('Random_Forest', ascending=False)

print("\nTop 10 Features by Random Forest Importance:")
print(importance_comparison.head(10).to_string(index=False))

# Visualize feature importance comparison
plt.figure(figsize=(14, 8))

x_pos = np.arange(len(importance_comparison))
width = 0.35

plt.barh(x_pos - width/2, importance_comparison['Decision_Tree'], width, 
         label='Decision Tree', color='steelblue', alpha=0.8)
plt.barh(x_pos + width/2, importance_comparison['Random_Forest'], width, 
         label='Random Forest', color='forestgreen', alpha=0.8)

plt.yticks(x_pos, importance_comparison['Feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance Comparison: Decision Tree vs Random Forest', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(axis='x', alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("\n[SUCCESS] Feature importance comparison saved as 'feature_importance_comparison.png'")
plt.close()

# ============================================================================
# ACCURACY COMPARISON VISUALIZATION
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart comparison
models = ['Decision Tree', 'Random Forest']
train_accs = [dt_train_acc, rf_train_acc]
test_accs = [dt_test_acc, rf_test_acc]

x = np.arange(len(models))
width = 0.35

ax1.bar(x - width/2, train_accs, width, label='Training', color='skyblue', alpha=0.8)
ax1.bar(x + width/2, test_accs, width, label='Testing', color='coral', alpha=0.8)

ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, (train, test) in enumerate(zip(train_accs, test_accs)):
    ax1.text(i - width/2, train + 0.02, f'{train:.3f}', ha='center', fontsize=9)
    ax1.text(i + width/2, test + 0.02, f'{test:.3f}', ha='center', fontsize=9)

# Line plot showing improvement
ax2.plot(['Decision Tree', 'Random Forest'], [dt_test_acc, rf_test_acc], 
         'o-', linewidth=2, markersize=10, color='green')
ax2.axhline(y=dt_test_acc, color='red', linestyle='--', alpha=0.5, 
            label=f'Decision Tree Baseline ({dt_test_acc:.4f})')
ax2.fill_between(['Decision Tree', 'Random Forest'], dt_test_acc, rf_test_acc, 
                 alpha=0.2, color='green')
ax2.set_ylabel('Testing Accuracy', fontsize=12)
ax2.set_title('Accuracy Improvement', fontsize=14, fontweight='bold')
ax2.set_ylim(0.95, 1.0)
ax2.grid(alpha=0.3)
ax2.legend()

# Add improvement annotation
ax2.annotate(f'+{improvement:.4f}\n({improvement_pct:+.2f}%)', 
             xy=(1, rf_test_acc), xytext=(0.5, rf_test_acc + 0.01),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=10, ha='center', color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Accuracy comparison saved as 'accuracy_comparison.png'")
plt.close()

# ============================================================================
# N_ESTIMATORS ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("RANDOM FOREST - N_ESTIMATORS ANALYSIS")
print("=" * 70)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, rf_scores, 'o-', linewidth=2, markersize=10, color='forestgreen')
plt.axhline(y=dt_test_acc, color='steelblue', linestyle='--', 
            label=f'Decision Tree Baseline ({dt_test_acc:.4f})')
plt.axvline(x=best_n_estimators, color='red', linestyle='--', 
            label=f'Best ({best_n_estimators})')
plt.xlabel('Number of Estimators', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Random Forest Performance vs Number of Trees', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('random_forest_n_estimators_analysis.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] N_estimators analysis saved as 'random_forest_n_estimators_analysis.png'")
plt.close()

# Print n_estimators results
for n_est, score in zip(n_estimators_list, rf_scores):
    print(f"n_estimators={n_est:3d}: {score:.4f} ({score*100:.2f}%)")
print(f"Best: n_estimators={best_n_estimators} with {max(rf_scores):.4f} ({max(rf_scores)*100:.2f}%)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON SUMMARY")
print("=" * 70)

print(f"\n{'Metric':<30} {'Decision Tree':<20} {'Random Forest':<20}")
print("-" * 70)
print(f"{'Training Accuracy':<30} {dt_train_acc*100:>7.2f}%           {rf_train_acc*100:>7.2f}%")
print(f"{'Testing Accuracy':<30} {dt_test_acc*100:>7.2f}%           {rf_test_acc*100:>7.2f}%")
print(f"{'Overfitting Gap':<30} {(dt_train_acc-dt_test_acc)*100:>7.2f}%           {(rf_train_acc-rf_test_acc)*100:>7.2f}%")
print(f"{'Improvement':<30} {'-'*18} {'+' if improvement > 0 else ''}{abs(improvement)*100:>6.2f}%")

# Determine winner
winner = "Random Forest" if rf_test_acc > dt_test_acc else "Decision Tree"
print(f"\nWinner: {winner}")
print(f"Best Accuracy: {max(dt_test_acc, rf_test_acc):.4f} ({max(dt_test_acc, rf_test_acc)*100:.2f}%)")

print("\n" + "=" * 70)
print("[SUCCESS] ALL COMPARISONS COMPLETE!")
print("=" * 70)

