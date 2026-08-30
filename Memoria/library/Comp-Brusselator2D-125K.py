import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# 1. Datos
# ============================

data = [
    ["Runge Kutta", "Secuencial", 2531.01, 1],
    ["Runge Kutta", "OMP",        236.42,  10.70556636],
    ["Runge Kutta", "CUDA",       77.1447, 32.80860513],

    ["Adams Bashford", "Secuencial", 847.87, 1],
    ["Adams Bashford", "OMP",        82.704, 10.25186206],
    ["Adams Bashford", "CUDA",       23.5544, 35.99624699],

    ["Adams Bashford Moulton", "Secuencial", 2692.79, 1],
    ["Adams Bashford Moulton", "OMP",        264.402, 10.18445398],
    ["Adams Bashford Moulton", "CUDA",       86.2916, 31.20570252],
]

df = pd.DataFrame(data, columns=["Solver", "Tipo", "T", "Speedup"])

# ============================
# 2. Orden manual de grupos
# ============================

solvers = ["Runge Kutta", "Adams Bashford", "Adams Bashford Moulton"]
df["Solver"] = pd.Categorical(df["Solver"], categories=solvers, ordered=True)
df = df.sort_values(["Solver", "Tipo"])

# ============================
# 3. Parámetros de espaciado
# ============================

bar_width = 0.35
pair_spacing = 0.05
intra_group_spacing = 0.15
inter_group_spacing = 0.6

# ============================
# 4. Preparar la gráfica
# ============================

tipos = ["Secuencial", "OMP", "CUDA"]

color_tiempo = "#1f77b4"     # azul
color_speedup = "#ff7f0e"    # naranja

fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()  # segundo eje Y para speedup

offset = 0
xticks = []
xtick_labels = []
group_centers = []

for solver in solvers:
    df_s = df[df["Solver"] == solver]

    group_start = offset

    for tipo in tipos:
        fila = df_s[df_s["Tipo"] == tipo].iloc[0]

        # Barra de tiempo (eje Y izquierdo)
        ax1.bar(offset,
                fila["T"],
                width=bar_width,
                color=color_tiempo)

        # Barra de speedup (eje Y derecho)
        ax2.bar(offset + bar_width + pair_spacing,
                fila["Speedup"],
                width=bar_width,
                color=color_speedup)

        pareja_center = offset + (bar_width + pair_spacing) / 2
        xticks.append(pareja_center)
        xtick_labels.append(tipo)

        offset += (2 * bar_width + pair_spacing) + intra_group_spacing

    group_end = offset - intra_group_spacing
    group_centers.append((group_start + group_end) / 2)

    offset += inter_group_spacing

# ============================
# 5. Líneas horizontales suaves
# ============================

ax1.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

# ============================
# 6. Leyenda arriba a la derecha dentro del recuadro
# ============================

handles = [
    plt.Rectangle((0,0),1,1, color=color_tiempo),
    plt.Rectangle((0,0),1,1, color=color_speedup)
]
labels = ["Tiempo (s)", "Speedup"]

ax1.legend(handles, labels,
           fontsize=15,
           loc="upper right")

# ============================
# 7. Etiquetas y presentación
# ============================

ax1.set_ylabel("Tiempo (s)", fontsize=15)
ax2.set_ylabel("Speedup", fontsize=15)

ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels, fontsize=15)

# Etiquetas de grupo arriba
for center, solver in zip(group_centers, solvers):
    ax1.text(center, ax1.get_ylim()[1] * 1.02, solver,
             ha='center', va='bottom', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
