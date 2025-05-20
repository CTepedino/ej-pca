import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import sys

original_stdout = sys.stdout
f = open('results.txt', 'w', encoding='utf-8')
sys.stdout = f

df = pd.read_csv('europe.csv')

X = df.drop('Country', axis=1)
countries = df['Country']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    data=X_pca,
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
)
pca_df['Country'] = countries

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), 
         cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance')
plt.grid(True)
plt.ylim(0, 1.1)  
plt.savefig('cumulative_variance.png')
plt.close()

plt.figure(figsize=(12, 6))
feature_contributions = pd.DataFrame(
    pca.components_[0],
    index=X.columns,
    columns=['PC1']
)
feature_contributions_sorted = feature_contributions.sort_values('PC1', ascending=False)
plt.bar(range(len(feature_contributions_sorted)), feature_contributions_sorted['PC1'])
plt.xticks(range(len(feature_contributions_sorted)), feature_contributions_sorted.index, rotation=45, ha='right')
plt.ylabel('Contribución a PC1')
plt.title('Contribución de cada variable al Primer Componente Principal')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('pc1_contributions.png')
plt.close()

# Create bar plot for PC1 scores by country
plt.figure(figsize=(15, 8))
pc1_scores = pd.DataFrame({'Country': countries, 'PC1': X_pca[:, 0]})
pc1_scores_sorted = pc1_scores.sort_values('PC1', ascending=False)
plt.bar(range(len(pc1_scores_sorted)), pc1_scores_sorted['PC1'])
plt.xticks(range(len(pc1_scores_sorted)), pc1_scores_sorted['Country'], rotation=45, ha='right')
plt.ylabel('Score PC1')
plt.title('Contribución del PC1 por País')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('pc1_countries.png')
plt.close()

# Create biplot for PC1 vs PC2
plt.figure(figsize=(12, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'])

# Add country labels
for i, country in enumerate(countries):
    plt.annotate(country, (pca_df['PC1'][i], pca_df['PC2'][i]))

# Add arrows for each variable
for i, feature in enumerate(X.columns):
    plt.arrow(0, 0, pca.components_[0][i]*3, pca.components_[1][i]*3, 
              color='r', alpha=0.5, head_width=0.1)
    plt.text(pca.components_[0][i]*3.2, pca.components_[1][i]*3.2, 
             feature, color='r', ha='center', va='center')

plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} varianza explicada)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} varianza explicada)')
plt.title('Biplot de Países Europeos')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('pca_biplot.png')
plt.close()

print("\nRatio de varianza explicada para cada componente:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.2%}")

print("\nVarianza acumulada:")
for i, ratio in enumerate(cumulative_variance_ratio):
    print(f"PC{i+1}: {ratio:.2%}")

f.close()
sys.stdout = original_stdout

print("El análisis se ha completado. Los resultados se han guardado en 'results.txt'") 