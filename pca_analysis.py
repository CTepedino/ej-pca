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
plt.savefig('cumulative_variance.png')
plt.close()

# Create a biplot for PC1 vs PC2
plt.figure(figsize=(12, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'])

for i, country in enumerate(countries):
    plt.annotate(country, (pca_df['PC1'][i], pca_df['PC2'][i]))

plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} variance explained)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} variance explained)')
plt.title('PCA Biplot of European Countries')
plt.grid(True)
plt.savefig('pca_biplot.png')
plt.close()

feature_contributions = pd.DataFrame(
    pca.components_[0],
    index=X.columns,
    columns=['PC1']
)
print("\nContribuciones de las variables a PC1:")
print(feature_contributions.sort_values('PC1', ascending=False))

print("\nRatio de varianza explicada para cada componente:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.2%}")

print("\nVarianza acumulada:")
for i, ratio in enumerate(cumulative_variance_ratio):
    print(f"PC{i+1}: {ratio:.2%}")

# Close the file and restore stdout
f.close()
sys.stdout = original_stdout

print("El análisis se ha completado. Los resultados se han guardado en 'results.txt'") 