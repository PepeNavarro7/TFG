import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse


def view(filename):
    x_vals = []
    y_vals = []
    z_vals = []
    matrix_vals = []  # Para el caso D

    tipo = None  # A, B o D

    # -------------------------
    #   LECTURA DEL ARCHIVO
    # -------------------------
    with open(filename, 'r') as file:
        for linea in file:
            datos = linea.split()

            # Detectar tipo en la primera línea válida
            if tipo is None:
                if len(datos) == 2:
                    tipo = "A"
                elif len(datos) == 3:
                    tipo = "B"
                elif len(datos) > 3:
                    tipo = "D"   # Ya no existe el caso C
                else:
                    print(f"Línea ignorada por formato incorrecto: {linea.strip()}")
                    continue

            # Procesar según tipo
            if tipo == "A" and len(datos) == 2:
                x_vals.append(float(datos[0]))
                y_vals.append(float(datos[1]))

            elif tipo == "B" and len(datos) == 3:
                x_vals.append(float(datos[0]))
                y_vals.append(float(datos[1]))
                z_vals.append(float(datos[2]))

            elif tipo == "D" and len(datos) > 3:
                fila = list(map(float, datos))
                x_vals.append(fila[0])
                matrix_vals.append(fila)

    # -------------------------
    #   CASO A (2 columnas)
    # -------------------------
    if tipo == "A":
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='f(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(filename)
        plt.legend()
        plt.grid(True)
        plt.savefig("plot_" + filename + ".png")
        print("Gráfica guardada.")
        return

    # -------------------------
    #   CASO B (3 columnas)
    # -------------------------
    if tipo == "B":
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='f1(x)')
        plt.plot(x_vals, z_vals, marker='o', linestyle='-', color='r', label='f2(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(filename)
        plt.legend()
        plt.grid(True)
        plt.savefig("plot_" + filename + ".png")
        print("Gráfica guardada.")
        return

    # -------------------------
    #   CASO D → 2 superficies + 1 combinada
    # -------------------------
    matrix = np.array(matrix_vals)

    # Separar pares Y,Z
    Y1 = matrix[:, 1::2]   # columnas 1,3,5,...
    Z1 = matrix[:, 2::2]   # columnas 2,4,6,...

    X = np.array(x_vals)
    M = Y1.shape[1]
    Y_index = np.arange(M)

    X_grid, Y_grid = np.meshgrid(X, Y_index, indexing='ij')

    # Superficie 1
    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X_grid, Y_grid, Y1, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Índice Y')
    ax1.set_zlabel('Valor')
    ax1.set_title(filename + " — Superficie 1")
    plt.savefig(filename + "_surface1.png")

    # Superficie 2
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X_grid, Y_grid, Z1, cmap='plasma', edgecolor='none')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Índice Y')
    ax2.set_zlabel('Valor')
    ax2.set_title(filename + " — Superficie 2")
    plt.savefig(filename + "_surface2.png")

    # Superficie combinada
    fig3 = plt.figure(figsize=(10, 7))
    ax3 = fig3.add_subplot(111, projection='3d')

    ax3.plot_surface(X_grid, Y_grid, Y1, cmap='viridis', alpha=0.7, edgecolor='none')
    ax3.plot_surface(X_grid, Y_grid, Z1, cmap='plasma', alpha=0.7, edgecolor='none')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Índice Y')
    ax3.set_zlabel('Valor')
    ax3.set_title(filename + " — Superficies combinadas")

    plt.savefig(filename + "_surface_both.png")

    print("Superficies 3D guardadas como:")
    print("  → plot_" + filename + "_surface1.png")
    print("  → plot_" + filename + "_surface2.png")
    print("  → plot_" + filename + "_surface_both.png")



parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

view(args.filename)
