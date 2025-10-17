import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load preprocessed data
data_path = Path(__file__).parent.parent / 'data' / 'preprocessed_data.csv'
df = pd.read_csv(data_path)

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Separate features from any potential labels/identifiers
# Adjust this based on your actual data structure
# Assuming all numeric columns are features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Print explained variance
print("\nExplained variance ratio by component:")
for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
    print(f"PC{i}: {var:.4f} ({var*100:.2f}%)")

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(f"\nCumulative variance explained by first 2 components: {cumsum[1]:.4f}")
print(f"Cumulative variance explained by first 3 components: {cumsum[2]:.4f}")

# Visualize PCA results in 2D
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot of first two principal components
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, edgecolors='k', s=50)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
axes[0].set_title('PCA: First Two Principal Components')
axes[0].grid(True, alpha=0.3)

# Scree plot
axes[1].plot(range(1, len(pca.explained_variance_ratio_[:10])+1), 
             pca.explained_variance_ratio_[:10], 'bo-')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].set_title('Scree Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'results' / 'pca_2d_analysis.png', dpi=300)
plt.show()

# 3D visualization if applicable
if X_pca.shape[1] >= 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                        alpha=0.6, edgecolors='k', s=50)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)')
    ax.set_title('PCA: First Three Principal Components')
    plt.savefig(Path(__file__).parent.parent / 'results' / 'pca_3d_analysis.png', dpi=300)
    plt.show()

# Save PCA results
results_df = pd.DataFrame(
    X_pca[:, :3],
    columns=['PC1', 'PC2', 'PC3'],
    index=df.index
)
results_df.to_csv(Path(__file__).parent.parent / 'results' / 'pca_results.csv')

print("\nPCA analysis completed. Results saved.")
