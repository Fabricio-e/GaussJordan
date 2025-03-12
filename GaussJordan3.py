import numpy as np

def gauss_jordan_pivot_determinante(A, b):
    """
    Resuelve un sistema de ecuaciones Ax = b mediante el método de Gauss-Jordan con pivoteo parcial
    e imprime el determinante de A para verificar si el sistema tiene solución única.
    """
    n = len(A)
    # Matriz aumentada
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)
    
    # Cálculo del determinante de A
    det_A = np.linalg.det(A)
    
    # Verificar si el sistema es determinado o indeterminado
    if np.isclose(det_A, 0):
        print(f"Determinante de A: {det_A:.5f}. El sistema es indeterminado o no tiene solución única.")
        return None
    
    print(f"Determinante de A: {det_A:.5f}. El sistema tiene solución única.")
    
    # Aplicación del método de Gauss-Jordan con pivoteo
    for i in range(n):
        # Pivoteo parcial
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        # Normalización de la fila pivote
        Ab[i] = Ab[i] / Ab[i, i]

        # Eliminación en otras filas
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[j, i] * Ab[i]

    # Extraer la solución
    x = Ab[:, -1]
    return x

# Definir el sistema de ecuaciones de la imagen
A = np.array([[2, -3, 4, -1, 5, -1, 2, -1, 3, -2],
              [-3, 2, 5, -1, 4, 2, -3, 3, -2, 5],
              [4, -1, 3, 2, 4, -3, -2, 5, -4, 10],
              [-1, 5, -2, 3, 4, -5, 2, 3, 9, -5],
              [3, -2, 5, -1, 4, 2, -3, 7, -2, 10],
              [-2, 4, -3, 3, 5, -6, 2, -4, 3, -10],
              [5, -1, 2, -3, 4, -1, -2, 3, -1, 9],
              [-1, 3, -2, 4, 5, -6, 2, -1, 4, -6],
              [2, -3, 4, -1, 5, -6, 2, -3, 3, 10],
              [-3, 2, 4, -3, -5, 2, 5, 3, 9, -8]], dtype=float)

b = np.array([11, -10, 8, -6, 7, -10, 9, 5, 6, -8], dtype=float)

# Resolver el sistema
interaccion_termica_solucion = gauss_jordan_pivot_determinante(A, b)

# Imprimir la solución si existe
if interaccion_termica_solucion is not None:
    print("Solución del sistema:", interaccion_termica_solucion)