import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("=" * 80)
print("CROSS-VALIDATION EVALUATION - DECISION TREE vs RANDOM FOREST")
print("=" * 80)

df = pd.read_csv('heart.csv')
print(f"\nDataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Check data distribution
print(f"\nTarget distribution:")
print(y.value_counts())

# ============================================================================
# CROSS-VALIDATION WITH DIFFERENT FOLD SIZES
# ============================================================================
folds_list = [3, 5, 10]
results_dt = {}
results_rf = {}

print("\n" + "=" * 80)
print("CROSS-VALIDATION PERFORMANCE")
print("=" * 80)

for n_folds in folds_list:
    print(f"\n{n_folds}-Fold Cross-Validation:")
    print("-" * 80)
    
    # Use stratified k-fold for balanced splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=42)
    dt_scores = cross_val_score(dt_model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    results_dt[n_folds] = {
        'scores': dt_scores,
        'mean': dt_scores.mean(),
        'std': dt_scores.std(),
        'min': dt_scores.min(),
        'max': dt_scores.max()
    }
    
    print(f"Decision Tree:")
    print(f"  Fold Accuracies: {[f'{s:.4f}' for s in dt_scores]}")
    print(f"  Mean ± Std: {dt_scores.mean():.4f} ± {dt_scores.std():.4f}")
    print(f"  Range: [{dt_scores.min():.4f}, {dt_scores.max():.4f}]")
    print(f"  CV Score: {dt_scores.mean():.4f} ± {dt_scores.std():.4f}")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, 
                                     criterion='entropy', random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf_model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    results_rf[n_folds] = {
        'scores': rf_scores,
        'mean': rf_scores.mean(),
        'std': rf_scores.std(),
        'min': rf_scores.min(),
        'max': rf_scores.max()
    }
    
    print(f"\nRandom Forest:")
    print(f"  Fold Accuracies: {[f'{s:.4f}' for s in rf_scores]}")
    print(f"  Mean ± Std: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    print(f"  Range: [{rf_scores.min():.4f}, {rf_scores.max():.4f}]")
    print(f"  CV Score: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

# ============================================================================
# DETAILED 10-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED 10-FOLD CROSS-VALIDATION")
print("=" * 80)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
dt_fold_scores = []
rf_fold_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=42)
    dt_model.fit(X_train_fold, y_train_fold)
    dt_accuracy = dt_model.score(X_test_fold, y_test_fold)
    dt_fold_scores.append(dt_accuracy)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, 
                                     criterion='entropy', random_state=42, n_jobs=-1)
    rf_model.fit(X_train_fold, y_train_fold)
    rf_accuracy = rf_model.score(X_test_fold, y_test_fold)
    rf_fold_scores.append(rf_accuracy)

dt_fold_scores = np.array(dt_fold_scores)
rf_fold_scores = np.array(rf_fold_scores)

print("\nFold-by-Fold Results:")
print(f"{'Fold':<8} {'Decision Tree':<20} {'Random Forest':<20}")
print("-" * 50)
for i in range(10):
    print(f"Fold {i+1:<4} {dt_fold_scores[i]:.4f}             {rf_fold_scores[i]:.4f}")

print("\nSummary Statistics:")
print(f"{'Metric':<20} {'Decision Tree':<20} {'Random Forest':<20}")
print("-" * 60)
print(f"{'Mean Accuracy':<20} {dt_fold_scores.mean():.4f}         {rf_fold_scores.mean():.4f}")
print(f"{'Std Deviation':<20} {dt_fold_scores.std():.4f}         {rf_fold_scores.std():.4f}")
print(f"{'Min Accuracy':<20} {dt_fold_scores.min():.4f}         {rf_fold_scores.min():.4f}")
print(f"{'Max Accuracy':<20} {dt_fold_scores.max():.4f}         {rf_fold_scores.max():.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy distribution across folds (10-fold)
ax1 = axes[0, 0]
x_pos = np.arange(10) + 1
width = 0.35
ax1.bar(x_pos - width/2, dt_fold_scores, width, label='Decision Tree', 
        color='steelblue', alpha=0.8)
ax1.bar(x_pos + width/2, rf_fold_scores, width, label='Random Forest', 
        color='forestgreen', alpha=0.8)
ax1.set_xlabel('Fold Number', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_title('10-Fold Cross-Validation: Accuracy per Fold', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'Fold {i}' for i in range(1, 11)], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0.90, 1.0)

# Add mean lines
ax1.axhline(y=dt_fold_scores.mean(), color='blue', linestyle='--', alpha=0.5)
ax1.axhline(y=rf_fold_scores.mean(), color='green', linestyle='--', alpha=0.5)

# 2. Box plot comparison
ax2 = axes[0, 1]
data_to_plot = [dt_fold_scores, rf_fold_scores]
bp = ax2.boxplot(data_to_plot, labels=['Decision Tree', 'Random Forest'],
                 patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('forestgreen')
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Distribution of Cross-Validation Scores', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Variance across folds
ax3 = axes[1, 0]
folds = [3, 5, 10]
dt_means = [results_dt[f]['mean'] for f in folds]
dt_stds = [results_dt[f]['std'] for f in folds]
rf_means = [results_rf[f]['mean'] for f in folds]
rf_stds = [results_rf[f]['std'] for f in folds]

ax3.plot(folds, dt_means, 'o-', label='Decision Tree', linewidth=2, markersize=10, color='steelblue')
ax3.fill_between(folds, 
                [m - s for m, s in zip(dt_means, dt_stds)],
                [m + s for m, s in zip(dt_means, dt_stds)],
                alpha=0.2, color='steelblue')
ax3.plot(folds, rf_means, 's-', label='Random Forest', linewidth=2, markersize=10, color='forestgreen')
ax3.fill_between(folds,
                [m - s for m, s in zip(rf_means, rf_stds)],
                [m + s for m, s in zip(rf_means, rf_stds)],
                alpha=0.2, color='forestgreen')
ax3.set_xlabel('Number of Folds', fontsize=11)
ax3.set_ylabel('Mean CV Accuracy', fontsize=11)
ax3.set_title('Cross-Validation Stability Across Fold Sizes', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Standard deviation comparison
ax4 = axes[1, 1]
ax4.bar(['Decision Tree', 'Random Forest'], 
        [dt_fold_scores.std(), rf_fold_scores.std()],
        color=['steelblue', 'forestgreen'], alpha=0.8)
ax4.set_ylabel('Standard Deviation', fontsize=11)
ax4.set_title('Model Stability (Lower is Better)', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for i, (name, std) in enumerate(zip(['Decision Tree', 'Random Forest'], 
                                   [dt_fold_scores.std(), rf_fold_scores.std()])):
    ax4.text(i, std + 0.001, f'{std:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('cross_validation_analysis.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Cross-validation analysis saved as 'cross_validation_analysis.png'")
plt.close()

# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

from scipy import stats

# Perform t-test to compare means
t_statistic, p_value = stats.ttest_rel(dt_fold_scores, rf_fold_scores)

print(f"\nPaired t-test (Decision Tree vs Random Forest):")
print(f"  t-statistic: {t_statistic:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significance: {'Significant' if p_value < 0.05 else 'Not Significant'}")
print(f"  Difference: {rf_fold_scores.mean() - dt_fold_scores.mean():.4f}")

# Confidence intervals (95%)
def confidence_interval(data):
    mean = data.mean()
    std = data.std()
    n = len(data)
    se = std / np.sqrt(n)
    t_critical = 2.262  # for 95% CI with 9 degrees of freedom (10 folds)
    margin = t_critical * se
    return mean, margin

dt_mean, dt_margin = confidence_interval(dt_fold_scores)
rf_mean, rf_margin = confidence_interval(rf_fold_scores)

print(f"\n95% Confidence Intervals:")
print(f"  Decision Tree: [{dt_mean - dt_margin:.4f}, {dt_mean + dt_margin:.4f}]")
print(f"  Random Forest: [{rf_mean - rf_margin:.4f}, {rf_mean + rf_margin:.4f}]")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("CROSS-VALIDATION SUMMARY")
print("=" * 80)

print(f"\n{'Model':<20} {'Mean CV Score':<20} {'Std Dev':<15} {'95% CI'}")
print("-" * 75)
print(f"{'Decision Tree':<20} {dt_fold_scores.mean():.4f}             "
      f"{dt_fold_scores.std():.4f}         [{dt_mean - dt_margin:.4f}, {dt_mean + dt_margin:.4f}]")
print(f"{'Random Forest':<20} {rf_fold_scores.mean():.4f}             "
      f"{rf_fold_scores.std():.4f}         [{rf_mean - rf_margin:.4f}, {rf_mean + rf_margin:.4f}]")

print(f"\n{'Model Stability (CV Std Dev)':<30} {'Winner'}")
print("-" * 50)
if dt_fold_scores.std() < rf_fold_scores.std():
    print(f"{'Decision Tree is more stable':<30} Decision Tree")
else:
    print(f"{'Random Forest is more stable':<30} Random Forest")

print(f"\n{'Mean Accuracy':<30} {'Winner'}")
print("-" * 50)
if dt_fold_scores.mean() > rf_fold_scores.mean():
    print(f"{dt_fold_scores.mean():.4f}: Decision Tree higher               Decision Tree")
elif rf_fold_scores.mean() > dt_fold_scores.mean():
    print(f"{rf_fold_scores.mean():.4f}: Random Forest higher              Random Forest")
else:
    print("Tie - Both models have identical performance")

# Overall assessment
print("\n" + "=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)

if dt_fold_scores.mean() > rf_fold_scores.mean() and dt_fold_scores.std() < rf_fold_scores.std():
    winner = "Decision Tree - better accuracy AND more stable"
elif rf_fold_scores.mean() > dt_fold_scores.mean() and rf_fold_scores.std() < dt_fold_scores.std():
    winner = "Random Forest - better accuracy AND more stable"
elif dt_fold_scores.std() < rf_fold_scores.std():
    winner = "Decision Tree - more stable (similar accuracy)"
else:
    winner = "Random Forest - more stable (similar accuracy)"

print(f"\nOverall Winner: {winner}")
print(f"Stability: {'Low variance' if min(dt_fold_scores.std(), rf_fold_scores.std()) < 0.01 else 'Moderate variance'}")
print(f"Both models show {'excellent' if min(dt_fold_scores.mean(), rf_fold_scores.mean()) > 0.95 else 'good'} generalization")

print("\n" + "=" * 80)
print("[SUCCESS] CROSS-VALIDATION EVALUATION COMPLETE!")
print("=" * 80)

