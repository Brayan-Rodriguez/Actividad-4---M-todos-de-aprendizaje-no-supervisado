
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Crear dataset sintético
np.random.seed(42)
num_registros = 1000

dataset = pd.DataFrame({
    "bus_id": np.random.choice(["Bus_01", "Bus_02", "Bus_03", "Bus_04"], num_registros),
    "hora_dia": np.random.randint(5, 23, num_registros),
    "latitud": np.random.uniform(4.60, 4.75, num_registros),   # Ejemplo en Bogotá
    "longitud": np.random.uniform(-74.15, -74.05, num_registros),
    "velocidad_kmph": np.random.normal(30, 10, num_registros),
    "dia_semana": np.random.choice(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"], num_registros)
})

dataset.to_csv("fuente_datos_transporte.csv", index=False)

# Cargar dataset
dataset = pd.read_csv("fuente_datos_transporte.csv")

# Preprocesamiento
datos_numericos = dataset[["hora_dia", "latitud", "longitud", "velocidad_kmph"]]
escalador = StandardScaler()
datos_escalados = escalador.fit_transform(datos_numericos)

# Modelo de clustering
kmeans = KMeans(n_clusters=4, random_state=42)
dataset["cluster"] = kmeans.fit_predict(datos_escalados)

# Visualización
plt.figure(figsize=(10, 6))
plt.scatter(dataset["longitud"], dataset["latitud"], c=dataset["cluster"], cmap="viridis", alpha=0.5)
plt.title("Agrupación de rutas por zonas geográficas")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.savefig("clustering_resultados.png")
plt.show()
