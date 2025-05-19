import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def read_dataset(filename):
    with open(filename, "r") as f:
        f.readline()
        reader = csv.reader(f)
        data = []
        for line in reader:

            data.append({
                "Country": line[0],
                "Area": int(line[1]),
                "GDP": int(line[2]),
                "Inflation": float(line[3]),
                "Life.expect": float(line[4]),
                "Military": float(line[5]),
                "Pop.growth": float(line[6]),
                "Unemployment": float(line[7])
            })

    return data

if __name__ == "__main__":
    # Cargar los datos
    df = pd.read_csv("europe.csv")

    # Guardamos los nombres de los países
    countries = df["Country"]

    # Seleccionamos las variables numéricas para PCA
    features = ["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]
    X = df[features]

    # Estandarizamos los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # 6. Crear un DataFrame con resultados de PCA
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(len(features))])
    df_pca["Country"] = countries

    # 7. Visualizar países en PC1 vs PC2
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2")

    for i in range(df_pca.shape[0]):
        plt.text(df_pca.loc[i, "PC1"] + 0.1, df_pca.loc[i, "PC2"], df_pca.loc[i, "Country"], fontsize=8)

    plt.title("PCA de países europeos")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

    # 8. Mostrar la contribución de cada variable a PC1
    pc1_loadings = pd.Series(pca.components_[0], index=features)
    pc1_loadings.sort_values(ascending=False).plot(kind='bar', figsize=(8, 5), title="Contribución de variables a PC1")
    plt.ylabel("Peso")
    plt.grid(True)
    plt.show()

    # 9. Mostrar los valores numéricos de las cargas
    print("Cargas (pesos) de las variables en PC1:")
    print(pc1_loadings.sort_values(ascending=False))
