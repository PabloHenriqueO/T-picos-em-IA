import numpy as np

def coincidence_unordered(v1, v2):
  """
  Calcula o índice de coincidência não ordenada de vizinhanças.

  Argumentos:
    v1: Lista de elementos da primeira vizinhança.
    v2: Lista de elementos da segunda vizinhança.

  Retorno:
    O índice de coincidência não ordenada de vizinhanças.
  """

  # Converte as vizinhanças para conjuntos.
  set_v1 = set(v1)
  set_v2 = set(v2)

  # Interseção dos conjuntos.
  intersection = set_v1 & set_v2

  # União dos conjuntos.
  union = set_v1 | set_v2

  # Retorna o índice de coincidência.
  return len(intersection) / len(union)

def coincidence_ordered(v1, v2):
  """
  Calcula o índice de coincidência ordenada de vizinhanças.

  Argumentos:
    v1: Lista de elementos da primeira vizinhança.
    v2: Lista de elementos da segunda vizinhança.

  Retorno:
    O índice de coincidência ordenada de vizinhanças.
  """

  # Converte as vizinhanças para arrays.
  arr_v1 = np.array(v1)
  arr_v2 = np.array(v2)

  # Calcula a distância de Levenshtein.
  distance = np.levenshtein(arr_v1, arr_v2)

  # Retorna o índice de coincidência.
  return 1 - distance / max(len(arr_v1), len(arr_v2))

# Exemplo de uso.
v1 = [1, 2, 3, 4, 5]
v2 = [2, 3, 4, 5, 6]

coincidence_unordered = coincidence_unordered(v1, v2)
print(f"Índice de coincidência não ordenada: {coincidence_unordered}")

coincidence_ordered = coincidence_ordered(v1, v2)
print(f"Índice de coincidência ordenada: {coincidence_ordered}")
