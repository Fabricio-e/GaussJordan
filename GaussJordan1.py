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
A = np.array([[2, 1, -1, 4, -2, 5, -3, 1],
              [-3, 2, 4, -1, 3, -2, 5, -1],
              [4, -1, 3, 2, -3, 2, -2, 5],
              [-1, 5, -2, 3, 4, -1, 2, 4],
              [3, 2, -5, 3, 4, 2, -3, 1],
              [-2, 4, -3, 3, 5, -6, 2, 6],
              [5, 1, 2, -3, 4, 3, -2, -3],
              [1, -3, 3, -2, 4, 5, -6, 2]], dtype=float)

b = np.array([10, -5, 8, 4, -7, 6, -3, 9], dtype=float)

# Resolver el sistema
distribucion_solucion = gauss_jordan_pivot_determinante(A, b)

# Imprimir la solución si existe
if distribucion_solucion is not None:
    print("Solución del sistema:", distribucion_solucion)