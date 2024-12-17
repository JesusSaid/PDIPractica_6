import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def seleccionar_imagen():
    while True:
        print("Selecciona una imagen:")
        print("1- Alto contraste")
        print("2- Bajo contraste")
        print("3- Alta iluminación")
        print("4- Baja iluminación")
        print("5- Salir")
        opcion = int(input("Ingresa el número de la opción deseada: "))

        if opcion == 1:
            return cv.imread('altoContraste.jpeg', 0)
        elif opcion == 2:
            return cv.imread('bajocontraste.jpg', 0)
        elif opcion == 3:
            return cv.imread('AltaIluminacion.jpg', 0)
        elif opcion == 4:
            return cv.imread('BajaIluminacion.png', 0)
        elif opcion == 5:
            return None
        else:
            print("Opción inválida. Por favor, intenta nuevamente.")

def seleccionar_filtro():
    while True:
        print("Selecciona un filtro:")
        print("1- Filtro pasa bajos Ideal")
        print("2- Filtro pasa bajos Butterworth")
        print("3- Filtro pasa bajos Gaussian")
        print("4- Filtro pasa altos Ideal")
        print("5- Filtro pasa altos Butterworth")
        print("6- Filtro pasa altos Gaussian")
        print("7- Salir")
        opcion = int(input("Ingresa el número de la opción deseada: "))

        if opcion in [1, 2, 3, 4, 5, 6]:
            return opcion
        elif opcion == 7:
            return None
        else:
            print("Opción inválida. Por favor, intenta nuevamente.")

def solicitar_parametros_butterworth():
    print("Ingresa los parámetros para el filtro Butterworth:")
    radio = float(input("- Radio de corte (D): "))
    orden = int(input("- Orden del filtro (n): "))
    return {'radio': radio, 'orden': orden}

def solicitar_parametros_generales():
    print("Ingresa los parámetros para el filtro:")
    radio = float(input("- Radio de corte (D): "))
    return {'radio': radio}

def create_ideal_low_pass_filter(width, height, d):
    lp_filter = np.zeros((height, width), np.float32)
    centre = (width / 2, height / 2)
    for i in range(width):
        for j in range(height):
            radius = np.sqrt((i - centre[0])**2 + (j - centre[1])**2)
            lp_filter[j, i] = 1 if radius <= d else 0
    return lp_filter

def create_butterworth_low_pass_filter(width, height, d, n):
    lp_filter = np.zeros((height, width), np.float32)
    centre = (width / 2, height / 2)
    for i in range(width):
        for j in range(height):
            radius = max(1, np.sqrt((i - centre[0])**2 + (j - centre[1])**2))
            lp_filter[j, i] = 1 / (1 + (radius / d)**(2 * n))
    return lp_filter

def create_gaussian_low_pass_filter(width, height, d):
    lp_filter = np.zeros((height, width), np.float32)
    centre = (width / 2, height / 2)
    for i in range(width):
        for j in range(height):
            radius = np.sqrt((i - centre[0])**2 + (j - centre[1])**2)
            lp_filter[j, i] = np.exp(-(radius**2) / (2 * (d**2)))
    return lp_filter

def create_ideal_high_pass_filter(width, height, d):
    hp_filter = np.ones((height, width), np.float32)
    centre = (width / 2, height / 2)
    for i in range(width):
        for j in range(height):
            radius = np.sqrt((i - centre[0])**2 + (j - centre[1])**2)
            if radius <= d:
                hp_filter[j, i] = 0
    return hp_filter

def create_butterworth_high_pass_filter(width, height, d, n):
    hp_filter = np.zeros((height, width), np.float32)
    centre = (width / 2, height / 2)
    for i in range(width):
        for j in range(height):
            radius = np.sqrt((i - centre[0])**2 + (j - centre[1])**2)
            hp_filter[j, i] = 1 / (1 + (d / max(radius, 1e-10))**(2 * n))
    return hp_filter

def create_gaussian_high_pass_filter(width, height, d):
    hp_filter = np.zeros((height, width), np.float32)
    centre = (width / 2, height / 2)
    for i in range(width):
        for j in range(height):
            radius = np.sqrt((i - centre[0])**2 + (j - centre[1])**2)
            hp_filter[j, i] = 1 - np.exp(-(radius**2) / (2 * (d**2)))
    return hp_filter

def aplicar_filtro(I, filtro, parametros):
    ancho = I.shape[1]
    alto = I.shape[0]
    n_alto = cv.getOptimalDFTSize(alto)
    n_ancho = cv.getOptimalDFTSize(ancho)

    pad_der = n_ancho - ancho
    pad_abajo = n_alto - alto
    nI = cv.copyMakeBorder(I, 0, pad_abajo, 0, pad_der, cv.BORDER_CONSTANT, value=0)

    dft = cv.dft(np.float32(nI), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    if filtro == 1:
        filtro_espectral = create_ideal_low_pass_filter(n_ancho, n_alto, parametros['radio'])
    elif filtro == 2:
        filtro_espectral = create_butterworth_low_pass_filter(n_ancho, n_alto, parametros['radio'], parametros['orden'])
    elif filtro == 3:
        filtro_espectral = create_gaussian_low_pass_filter(n_ancho, n_alto, parametros['radio'])
    elif filtro == 4:
        filtro_espectral = create_ideal_high_pass_filter(n_ancho, n_alto, parametros['radio'])
    elif filtro == 5:
        filtro_espectral = create_butterworth_high_pass_filter(n_ancho, n_alto, parametros['radio'], parametros['orden'])
    elif filtro == 6:
        filtro_espectral = create_gaussian_high_pass_filter(n_ancho, n_alto, parametros['radio'])

    filtro_espectral = np.stack((filtro_espectral, filtro_espectral), axis=-1)
    filtered_dft = cv.mulSpectrums(dft_shift, filtro_espectral, flags=0)
    filtered_dft = np.fft.ifftshift(filtered_dft)
    filtered_img = cv.idft(filtered_dft, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)

    # Recortar para que coincida con el tamaño original
    filtered_img = filtered_img[:alto, :ancho]
    filtered_img_normalized = cv.normalize(filtered_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Sumar la imagen original para filtros pasa altos
    if filtro in [4, 5, 6]:
        filtered_img_normalized = cv.add(np.uint8(filtered_img_normalized), I)

    return np.uint8(filtered_img_normalized)

def main():
    while True:
        I = seleccionar_imagen()
        if I is None:
            print("Saliendo del programa...")
            break

        filtro = seleccionar_filtro()
        if filtro is None:
            print("Saliendo del programa...")
            break

        # Solicitar parámetros según el filtro seleccionado
        if filtro in [2, 5]:
            parametros = solicitar_parametros_butterworth()
        else:
            parametros = solicitar_parametros_generales()

        imagen_filtrada = aplicar_filtro(I, filtro, parametros)

        plt.subplot(1, 2, 1)
        plt.imshow(I, cmap='gray')
        plt.title("Imagen Original")

        plt.subplot(1, 2, 2)
        plt.imshow(imagen_filtrada, cmap='gray')
        plt.title("Imagen Filtrada")

        plt.show()

if __name__ == "__main__":
    main()
