import cv2
import numpy as np
import time
from numba import jit,prange
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# ======================= Función para Generar Máscara Laplaciana =======================
def generar_mascara_laplaciana(tamaño):
    if tamaño % 2 == 0 or tamaño < 3:
        raise ValueError("La máscara debe ser impar y >= 3")

    mascara = -1 * np.ones((tamaño, tamaño), dtype=np.float32)
    centro = tamaño // 2
    mascara[centro, centro] = (tamaño * tamaño) - 1
    return mascara

# ======================= Leer Imagen =======================
imagen = cv2.imread('imagenArbol.jpg', cv2.IMREAD_GRAYSCALE)
if imagen is None:
    raise FileNotFoundError("No se encontró la imagen 'imagen.jpg'.")

imagen_float = imagen.astype(np.float32)

# ======================= Parámetro de Tamaño de Máscara =======================
tamano_mascara = 21  # Cambia aquí a 3, 5, 7, etc.
laplaciana = generar_mascara_laplaciana(tamano_mascara)

# ======================= Filtro Numba =======================
@jit(nopython=True, parallel=True)
def filtro_laplaciano_numba(imagen, mascara):
    alto, ancho = imagen.shape
    offset = mascara.shape[0] // 2
    salida = np.zeros_like(imagen)

    for y in prange(offset, alto - offset):  # Paraleliza por filas
        for x in range(offset, ancho - offset):
            valor = 0.0
            for i in range(-offset, offset + 1):
                for j in range(-offset, offset + 1):
                    valor += imagen[y + i, x + j] * mascara[i + offset, j + offset]

            # Reemplazo manual de np.clip(valor, 0, 255)
            if valor < 0.0:
                valor = 0.0
            elif valor > 255.0:
                valor = 255.0

            salida[y, x] = valor
    return salida


# ======================= Filtro GPU PyCUDA =======================
mod = SourceModule("""
__global__ void filtro_laplaciano(float *imagen, float *mascara, float *salida, int ancho, int alto, int offset, int tamano_mascara) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= offset && x < (ancho - offset) && y >= offset && y < (alto - offset)) {
        float valor = 0.0;
        for (int i = -offset; i <= offset; i++) {
            for (int j = -offset; j <= offset; j++) {
                int idx = (i + offset) * tamano_mascara + (j + offset);
                valor += imagen[(y + i) * ancho + (x + j)] * mascara[idx];
            }
        }
        salida[y * ancho + x] = fminf(fmaxf(valor, 0.0), 255.0);
    }
}
""")

filtro_laplaciano_gpu = mod.get_function("filtro_laplaciano")

# ======================= Procesamiento Numba =======================
start_numba = time.time()
resultado_numba = filtro_laplaciano_numba(imagen_float, laplaciana)
end_numba = time.time()
print(f"Tiempo CPU con Numba: {end_numba - start_numba:.6f} segundos")
cv2.imwrite('resultado_numba.jpg', resultado_numba.astype(np.uint8))

# ======================= Procesamiento PyCUDA =======================
altura, ancho = imagen.shape
salida_gpu = np.zeros_like(imagen_float)
laplaciana_flat = laplaciana.flatten()

imagen_gpu = cuda.mem_alloc(imagen_float.nbytes)
cuda.memcpy_htod(imagen_gpu, imagen_float)

salida_gpu_mem = cuda.mem_alloc(salida_gpu.nbytes)

mascara_gpu = cuda.mem_alloc(laplaciana_flat.nbytes)
cuda.memcpy_htod(mascara_gpu, laplaciana_flat)

block_size = (16, 16, 1)
grid_size = ((ancho + block_size[0] - 1) // block_size[0],
             (altura + block_size[1] - 1) // block_size[1])

offset = laplaciana.shape[0] // 2

start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()

filtro_laplaciano_gpu(imagen_gpu, mascara_gpu, salida_gpu_mem,
                      np.int32(ancho), np.int32(altura), np.int32(offset), np.int32(tamano_mascara),
                      block=block_size, grid=grid_size)

end_gpu.record()
end_gpu.synchronize()
gpu_time = start_gpu.time_till(end_gpu) / 1000.0
print(f"Tiempo GPU con PyCUDA: {gpu_time:.6f} segundos")

cuda.memcpy_dtoh(salida_gpu, salida_gpu_mem)
cv2.imwrite('resultado_gpu.jpg', salida_gpu.astype(np.uint8))

# ======================= Mostrar Máscara =======================
print(f"Máscara Laplaciana generada ({tamano_mascara}x{tamano_mascara}):\n{laplaciana}")