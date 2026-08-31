import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 1. Datos
# ============================

data = [
    ["Estándar", 200, 4.25118],
    ["Estándar", 400, 4.3825],
    ["Estándar", 1000, 4.42428],
    ["Estándar", 2000, 4.47256],
    ["Estándar", 4000, 4.50782],
    ["Estándar", 10000, 4.64262],
    ["Estándar", 20000, 6.07753],
    ["Estándar", 30000, 6.45839],
    ["Estándar", 40000, 7.65326],
    ["Estándar", 60000, 9.44472],
    ["Estándar", 80000, 10.9232],
    ["Estándar", 100000, 13.191],
    ["Estándar", 150000, 23.0747],
    ["Estándar", 200000, 30.9127],
    ["Estándar", 250000, 41.324],
    ["Estándar", 300000, 49.0773],
    ["Estándar", 350000, 58.7036],
    ["Estándar", 400000, 64.3164],
    ["Estándar", 500000, 78.4998],
    ["Estándar", 600000, 92.2143],
    ["Estándar", 700000, 108.716],
    ["Estándar", 800000, 123.306],
    ["Estándar", 900000, 137.599],
    ["Estándar", 1000000, 152.187],
    ["Estándar", 1200000, 180.576],
    ["Estándar", 1400000, 209.347],
    ["Estándar", 1500000, 227.02],
    ["Estándar", 1600000, 238.003],
    ["Estándar", 1800000, 267.296],
    ["Estándar", 2000000, 296.847],

    ["Shuffle", 200, 4.27395],
    ["Shuffle", 400, 4.47631],
    ["Shuffle", 1000, 4.48073],
    ["Shuffle", 2000, 4.49739],
    ["Shuffle", 4000, 4.55148],
    ["Shuffle", 10000, 4.71993],
    ["Shuffle", 20000, 6.194],
    ["Shuffle", 40000, 7.71357],
    ["Shuffle", 80000, 10.5485],
    ["Shuffle", 100000, 12.0613],
    ["Shuffle", 150000, 21.8507],
    ["Shuffle", 200000, 29.0348],
    ["Shuffle", 250000, 37.7658],
    ["Shuffle", 300000, 47.6186],
    ["Shuffle", 350000, 56.9246],
    ["Shuffle", 400000, 64.0661],
    ["Shuffle", 500000, 79.2419],
    ["Shuffle", 600000, 93.8084],
    ["Shuffle", 700000, 108.272],
    ["Shuffle", 800000, 122.247],
    ["Shuffle", 900000, 136.73],
    ["Shuffle", 1000000, 151.235],
    ["Shuffle", 1200000, 179.841],
    ["Shuffle", 1400000, 208.619],
    ["Shuffle", 1600000, 237.406],
    ["Shuffle", 1800000, 266.385],
    ["Shuffle", 2000000, 294.948],

    ["Grid", 200, 4.40917],
    ["Grid", 400, 4.42845],
    ["Grid", 1000, 4.45638],
    ["Grid", 2000, 4.5598],
    ["Grid", 4000, 4.5202],
    ["Grid", 10000, 4.68111],
    ["Grid", 20000, 6.26689],
    ["Grid", 40000, 7.82265],
    ["Grid", 80000, 10.6911],
    ["Grid", 100000, 12.2336],
    ["Grid", 150000, 22.0248],
    ["Grid", 200000, 29.6992],
    ["Grid", 250000, 38.7275],
    ["Grid", 300000, 47.1094],
    ["Grid", 350000, 55.1169],
    ["Grid", 400000, 62.2658],
    ["Grid", 500000, 79.3901],
    ["Grid", 600000, 94.2763],
    ["Grid", 700000, 108.879],
    ["Grid", 800000, 122.502],
    ["Grid", 900000, 137.174],
    ["Grid", 1000000, 151.697],
    ["Grid", 1200000, 180.08],
    ["Grid", 1400000, 209.053],
    ["Grid", 1600000, 237.448],
    ["Grid", 1800000, 266.49],
    ["Grid", 2000000, 295.242],
]

df = pd.DataFrame(data, columns=["Tipo", "neqn", "T"])

# ============================
# 2. Separar las tres curvas
# ============================

df_std   = df[df["Tipo"] == "Estándar"]
df_shuf  = df[df["Tipo"] == "Shuffle"]
df_grid  = df[df["Tipo"] == "Grid"]

# ============================
# 3. Preparar la gráfica
# ============================

fig, ax = plt.subplots(figsize=(14, 7))

# ============================
# 4. Líneas
# ============================

ax.plot(df_std["neqn"], df_std["T"],
        color="#1f77b4", linewidth=2.5, label="CUDA estándar")

ax.plot(df_shuf["neqn"], df_shuf["T"],
        color="#ff7f0e", linewidth=2.5, label="Shuffle")

ax.plot(df_grid["neqn"], df_grid["T"],
        color="#2ca02c", linewidth=2.5, label="Grid multidimensional")

# ============================
# 5. Escalas logarítmicas
# ============================

ax.set_xscale("log")
ax.set_yscale("log")

# ============================
# 6. Líneas horizontales suaves
# ============================

ax.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

# ============================
# 7. Leyenda centrada arriba dentro del recuadro
# ============================

ax.legend(fontsize=15,
          loc="upper center",
          bbox_to_anchor=(0.5, 1.0))

# ============================
# 8. Etiquetas y presentación
# ============================

ax.set_xlabel("neqn (log)", fontsize=15)
ax.set_ylabel("T (s) [log]", fontsize=15)

ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
