import pandas as pd
import matplotlib.pyplot as plt

# ============================
# 1. Datos
# ============================

data = [
    ["CUDA con graph", 1250,    15.6045],
    ["CUDA con graph", 5000,    15.3411],
    ["CUDA con graph", 11250,   15.9351],
    ["CUDA con graph", 20000,   22.6578],
    ["CUDA con graph", 31250,   23.8161],
    ["CUDA con graph", 45000,   31.6381],
    ["CUDA con graph", 61250,   39.9935],
    ["CUDA con graph", 80000,   47.0315],
    ["CUDA con graph", 101250,  69.774],
    ["CUDA con graph", 125000,  99.3],
    ["CUDA con graph", 180000,  147.217],
    ["CUDA con graph", 320000,  264.624],
    ["CUDA con graph", 500000,  392.972],
    ["CUDA con graph", 720000,  573.024],
    ["CUDA con graph", 980000,  772.954],

    ["CUDA sin graph", 1250,    15.7724],
    ["CUDA sin graph", 5000,    15.7544],
    ["CUDA sin graph", 11250,   16.1561],
    ["CUDA sin graph", 20000,   22.6586],
    ["CUDA sin graph", 31250,   23.6049],
    ["CUDA sin graph", 45000,   30.7777],
    ["CUDA sin graph", 61250,   39.1357],
    ["CUDA sin graph", 80000,   43.574],
    ["CUDA sin graph", 101250,  62.0139],
    ["CUDA sin graph", 125000,  86.2916],
    ["CUDA sin graph", 180000,  120.547],
    ["CUDA sin graph", 320000,  213.418],
    ["CUDA sin graph", 500000,  313.984],
    ["CUDA sin graph", 720000,  458.03],
    ["CUDA sin graph", 980000,  615.165],
]

df = pd.DataFrame(data, columns=["Tipo", "neqn", "T"])

# ============================
# 2. Separar las dos curvas
# ============================

df_graph     = df[df["Tipo"] == "CUDA con graph"]
df_nograph   = df[df["Tipo"] == "CUDA sin graph"]

# ============================
# 3. Preparar la gráfica
# ============================

fig, ax = plt.subplots(figsize=(14, 7))

# ============================
# 4. Líneas
# ============================

ax.plot(df_graph["neqn"], df_graph["T"],
        color="#1f77b4", linewidth=2.5, label="CUDA con graph")

ax.plot(df_nograph["neqn"], df_nograph["T"],
        color="#ff7f0e", linewidth=2.5, label="CUDA sin graph")

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

ax.set_xlabel("neqn [log]", fontsize=15)
ax.set_ylabel("T(s) [log]", fontsize=15)

ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
