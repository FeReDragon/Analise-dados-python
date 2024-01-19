# Importação de bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregamento do dataset Iris
iris = sns.load_dataset('iris')
print("Primeiras 5 linhas do dataset Iris:")
print(iris.head())

# Informações básicas sobre o dataset
print("\nDimensões do dataset:")
print(iris.shape)

print("\nEstatísticas básicas do dataset:")
print(iris.describe())

# Visualização de dados com pairplot
print("\nGerando visualizações...")
sns.pairplot(iris, hue='species')
plt.show()

# Análise básica: média das características por espécie
print("\nMédia das características por espécie:")
print(iris.groupby('species').mean())
