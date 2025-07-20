import os
import argparse
from glob import glob
import numpy as np
import cv2
from model3d import Model3D
import shutil

# Necesario para la correcta ejecucion del codigo en consola, no se puede ejecutar simplemente con el run.
# python main.py --test_path imgs_template_real\secuencia --models_path 3d_models --detector KEYPOINTS
# python main.py --test_path imgs_template_real\test --models_path 3d_models --detector KEYPOINTS


"""
Cálculo de la homografía H_t_img que va desde la imagen de la plantilla escaneada a la imagen de
entrada utilizando detección de puntos de interés y emparejamiento de los mismos mediante
comparación de descriptores.
"""
def ejercicio2_1_1(query_img, pts):
    
    # Uso de sift para detectar puntos clave
    kp2, des2 = sift.detectAndCompute(query_img,None)

    # Hacemos la búsqueda de coincidencias entre la plantilla y la imagen de entrada
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if (len(good) < 4):
        print("No hay suficientes coincidencias para encontrar la homografía")
        matchesMask = None
    else:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # Cálculo de la homografía
        htimg, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        dst = cv2.perspectiveTransform(pts,htimg)
        
        #query_img = cv2.polylines(query_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)  # Podemos dibujar el cuadrado en la imagen de entrada
        draw_params = dict(matchColor = (0,255,0), # Mostramos los matches en verde
                singlePointColor = None,
                matchesMask = matchesMask, # Mostramos únicamente los inliers
                flags = 2)

    # Dibujar los matches
    img_matches = cv2.drawMatches(template_img,kp1,query_img,kp2,good,None,**draw_params)
    img_matches = cv2.resize(img_matches, (1400, 900))

    return img_matches, htimg, matchesMask, dst

"""
Cálculo de la homografía H_w_img.
Esta transformación nos lleva de coordenadas en mm de las cuatro esquinas del
objeto real a las coordenadas en píxeles sobre la imagen de la plantilla escaneada
"""
def ejercicio2_1_2(query_img, pts, htimg, dst):
            
    img_temp = query_img.copy()
    hwt = cv2.getPerspectiveTransform(pts_mm, pts)
    img = cv2.polylines(img_temp, [np.int32(dst)],True,255,3, cv2.LINE_AA)
    hwimg = htimg @ hwt
    img = cv2.resize(img, (1400, 900))
    return hwt, hwimg, img

"""
Cálculo de R y t a partir de H_w_img y la matriz de intrínsecos K.
"""
def ejercicio2_1_3(K, hwimg):
    # Localizar el plano en la imagen y calcular R, t -> P
    K_inv = np.linalg.inv(K)
    H_est = K_inv @ hwimg
    
    h1 = H_est[:,0]
    h2 = H_est[:,1] 
    h3 = H_est[:,2]

    lamb = np.linalg.norm(h1) # ||h1|| == ||h2|| == lambda

    r1 = np.array(h1/lamb)
    r2 = np.array(h2/lamb)
    t = np.array(h3/lamb).reshape(1, -1).T

    r3 = np.cross(r1, r2) # Producto vectorial
    
    R = np.column_stack((r1, r2, r3))

    P = K @ np.hstack((R, t))

    return P

"""
Se pide pintar el sistema de referencia de la escena proyectando una línea sobre
cada eje X, Y, Z con el color adecuado (RGB, respectivamente para el XYZ).
Además, se debe mostrar el modelo 3D del cubo sobre la imagen de entrada en la posición estipulada.
"""
def ejercicio2_1_4(query_img, model, model_3d_file):
    # Mostrar los ejes del sistema de referencia de la plantilla sobre una copia de la imagen de entrada.
    
    plot_img = query_img.copy()

    L = 30

    points3D = np.array([[0, 0, 0, 1],
                        [L, 0, 0, 1] , # Eje x
                        [0, L, 0, 1] ,  # Eje y
                        [0, 0, L, 1]] # Eje z
                        )
    
    points2D = P @ points3D.T
    # Normalizar, dividir por la coordenada homogénea
    points2D /= points2D[2]

    # Dibujar los ejes en la imagen
    cv2.line(plot_img, tuple(points2D[:2, 0].astype(int)), tuple(points2D[:2, 1].astype(int)), (0, 0, 255), 5)  # Línea roja
    cv2.line(plot_img, tuple(points2D[:2, 0].astype(int)), tuple(points2D[:2, 2].astype(int)), (0, 255, 0), 5)  # Línea verde
    cv2.line(plot_img, tuple(points2D[:2, 0].astype(int)), tuple(points2D[:2, 3].astype(int)), (255, 0, 0), 5)  # Línea azul


    # Mostrar el modelo 3D del cubo sobre la imagen plot_img
    model.load_from_obj(model_3d_file)
    
    # El cubo se debe colocar en el centro del cuadrado objetivo
    model.scale(21.5)
    model.translate(np.array([[92.5,126.5,21.5]]))
    model.plot_on_image(plot_img, P)

    plot_img = cv2.resize(plot_img, (1400, 900))
    
    if args.test_path == "imgs_template_real\\test":
        # Guardar la imagen resultado en el directorio resultado_imgs
        pathSalida = os.path.join(dirSalida, os.path.basename(query_img_path))
        cv2.imwrite(pathSalida, plot_img)

    return plot_img

def make_red_mask(plot_marco_img):
            
    # Definir el rango a utilizar para crear la máscara en el espacio HSV
    red1l = np.array([0, 30, 100])
    red1u = np.array([8, 255, 255])
    red2l = np.array([165, 30, 100])
    red2u = np.array([179, 255, 255])

    mask1 = cv2.inRange(plot_marco_img, red1l, red1u)
    mask2 = cv2.inRange(plot_marco_img, red2l, red2u)
    # Crear la máscara roja combinando ambas
    red_mask = cv2.add(mask1, mask2)

    # Umbralizar la máscara para visualizar las áreas rojas que nos interesan
    _, red_mask_thresholded = cv2.threshold(red_mask, 1, 255, cv2.THRESH_BINARY)
    red_mask_thresholded = cv2.resize(red_mask_thresholded, (1400, 900))
    return red_mask, red_mask_thresholded

def crear_contornos(red_mask):
    # Encontrar contornos en la máscara roja
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar los contornos para quedarnos solo con los que tienen 4 vértices
    countors4vertices = []
    for contour in contours:
        epsilon = 0.02*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            countors4vertices.append((area, approx)) # Lo incluimos como una tupla (area, contorno)
    
    # Los dos contornos más grandes son los que nos interesan, ya que deberian ser los del marco
    countors4vertices = sorted(countors4vertices, key=lambda x: x[0], reverse=True)  # Ordenar por área

    contorno_exterior = countors4vertices[0][1]  # Contorno exterior
    contorno_interior = countors4vertices[1][1]  # Contorno interior

    contornos = [contorno_exterior, contorno_interior]
    return contornos

def nombrar_esquinas(contornos, plot_marco_img, hwimg):
    # Coordenadas físicas del marco en milímetros
    marco_fisico = np.array([
        [0, 0],         # Esquina superior izquierda
        [210, 0],       # Esquina superior derecha
        [210, 185],     # Esquina inferior derecha
        [0, 185]        # Esquina inferior izquierda
    ], dtype=np.float32)

    # Transformar las coordenadas físicas al sistema de la imagen
    marco_fisico_homog = np.hstack((marco_fisico, np.ones((4, 1))))  # Convertir a homogéneas
    marco_imagen = np.dot(hwimg, marco_fisico_homog.T).T
    marco_imagen /= marco_imagen[:, 2:]  # Normalizar coordenadas homogéneas
    marco_imagen = marco_imagen[:, :2]  # Convertir a 2D

    ordered_corners = []
    for contour in contornos:
        corners = contour.reshape(4, 2)

        # Emparejar las esquinas detectadas con las esquinas físicas proyectadas
        emparejadas = []
        for esquina_fisica in marco_imagen:
            distancias = [np.linalg.norm(esquina_detectada - esquina_fisica) for esquina_detectada in corners]
            indice_min = np.argmin(distancias)
            emparejadas.append((indice_min, esquina_fisica))

        # Ordenar las esquinas detectadas según el orden de las esquinas físicas
        emparejadas = sorted(emparejadas, key=lambda x: marco_imagen.tolist().index(x[1].tolist()))
        corners_ordenadas = [corners[par[0]] for par in emparejadas]

        # Cambiamos el orden de las esquinas "ordenadas" para que coincidan con el orden requerido en el enunciado
        corners_ordenadas[1], corners_ordenadas[3] = corners_ordenadas[3], corners_ordenadas[1]
        ordered_corners.append(corners_ordenadas)

   
    # Dibujar las esquinas en la imagen
    for corner_set in ordered_corners:
        for i, esquina in enumerate(corner_set):
            cv2.circle(plot_marco_img, tuple(esquina.astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(plot_marco_img, str(i), tuple(esquina.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)

    plot_marco_img = cv2.resize(plot_marco_img, (1400, 900))

    return plot_marco_img

"""
Función auxiliar para marcar las esquinas manualmente en la imagen con el objetivo de calcular el error
"""
def marcar_esquinas_manual(img, output_file):
    """
    Permite al usuario marcar las esquinas manualmente en la imagen y guarda las coordenadas en un archivo.
    """
    puntos = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            puntos.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.resize(img, (1400, 900))
            cv2.imshow("Marcar esquinas", img)

    print("Marque las 4 esquinas del marco rojo en el orden: superior izquierda, superior derecha, inferior derecha, inferior izquierda.")
    img = cv2.resize(img, (1600, 1200))
    cv2.imshow("Marcar esquinas", img)
    cv2.setMouseCallback("Marcar esquinas", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(puntos) != 4:
        print("Error: Debe marcar exactamente 4 esquinas.")
        return

    with open(output_file, "a") as f:
        for punto in puntos:
            f.write(f"{punto[0]} {punto[1]}\n")
        f.write("\n") 

    print(f"Coordenadas guardadas en {output_file}")

"""
Cargar todas las esquinas reales desde un archivo de texto.
"""
def cargar_todas_esquinas_gt(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    bloques = []
    for i in range(0, len(lines), 4):
        bloque = []
        for j in range(4):
            x, y = map(int, lines[i + j].split())
            bloque.append([x, y])
        bloques.append(np.array(bloque)) 
    return bloques

"""
Calcula el error normalizado entre las esquinas detectadas y las esquinas GT.
"""
def calcular_error_normalizado(detectadas, gt):

    distancias = np.linalg.norm(detectadas - gt, axis=1)
    error_medio = np.mean(distancias)

    # Calcular el perímetro del marco GT
    perimetro = np.sum(np.linalg.norm(np.roll(gt, -1, axis=0) - gt, axis=1))

    if perimetro == 0:
        return float('inf')  # evitar división por cero
    else:
        return error_medio / perimetro

gt_index = 0  
gt_bloques = cargar_todas_esquinas_gt("error.txt")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Crea y ejecuta un detector sobre las imágenes de test')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="KEYPOINTS", help='Nombre del detector a ejecutar')
    parser.add_argument(
        '--test_path', default="", help='Carpeta con las imágenes de test')
    parser.add_argument(
        '--models_path', default="", help='Carpeta con los modelos 3D (.obj)')
    args = parser.parse_args()
  


    # Definir el tipo de algoritmo de detección a utilizar
    print("Detector seleccionado " + args.detector)
    planar_localizer_name = args.detector
    
    # Cargar la imagen de la plantilla escaneada    
    template_img_path = os.path.join(args.test_path, "template_cropped.png")
    template_img = cv2.imread(template_img_path)
    if template_img is None:
        print("No puedo encontrar la imagen " + template_img_path)

    # Leer la matriz de intrínsecos de la cámara.
    K = np.loadtxt(os.path.join(args.test_path, "intrinsics.txt"))
    
    # Crear el detector de la plantilla pertinente (con KEYPOINTS u otro).
    # Inicializar detector SIFT
    sift = cv2.SIFT_create()

    # Crear modelo
    model = Model3D()

    kp1, des1 = sift.detectAndCompute(template_img,None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    h,w,color = template_img.shape
    
    # Definir los puntos de la plantilla tanto en píxeles como en milímetros
    pts = np.float32([ [0,0],[w-1,0],[w-1,h-1],[0,h-1] ]).reshape(-1,1,2)
    pts_mm = np.float32([ [0,0],[0,210],[185,210],[185,0] ]).reshape(-1,1,2)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    

    # Cargar el modelo 3D del cubo y colocarlo en el lugar pedido.
    model_3d_file = os.path.join(args.models_path, "cubo.obj")
    print("Cargando el modelo 3D " + model_3d_file)

    output_file = open("resultado.txt", "w")
    error_file = open("error.txt", "w")


    # Recorrer las imágenes en el directorio seleccionado y procesarlas.    
    print("Probando el detector " + args.detector + " en " + args.test_path)
    paths = sorted(glob(os.path.join(args.test_path, "*.jpg")))

    # Crear la carpeta resultado_imgs
    dirSalida = "resultado_imgs"

    if os.path.exists(dirSalida):
        for root, dirs, files in os.walk(dirSalida, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(dirSalida)  

    os.makedirs(dirSalida, exist_ok=True)

    for f in paths:
        query_img_path = f
        if not os.path.isfile(query_img_path):
            break
                
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            print("No puedo encontrar la imagen " + query_img_path)
            continue
        
            
        ###############################  Ejercicio 2.1.1 ###############################


        img_matches, htimg, matchesMask, dst =  ejercicio2_1_1(query_img, pts)
        #cv2.imshow("Emparejamientos", img_matches)
        #cv2.waitKey(1000)


        ###############################  Ejercicio 2.1.2 ###############################


        hwt, hwimg, img_homografia =  ejercicio2_1_2(query_img, pts, htimg, dst)


        # Las funciones anteriores son necesarias para el ejercicio 3,
        #  por lo que deben ejecutarse independientemente del detector.
        if args.detector == "KEYPOINTS":

            #cv2.imshow("Emparejamientos", img_homografia)
            #cv2.waitKey(1000)


            ###############################  Ejercicio 2.1.3 ###############################


            P = ejercicio2_1_3(K, hwimg)


            ###############################  Ejercicio 2.1.4 ###############################


            plot_img = ejercicio2_1_4(query_img, model, model_3d_file)

            # Mostrar el resultado en pantalla.
            cv2.imshow("3D info on images", plot_img)
            cv2.waitKey(1000)


        #################################  Ejercicio 3 #################################
        """
        Se pide modificar el cálculo de H_t_img para que se base en la detección de las cuatro
        esquinas exteriores del marco rojo que rodea al cubo en la imagen de entrada.
        """
        if args.detector == "THRESHOLD":
            # Detección de los bordes del marco rojo
            plot_marco_img = query_img.copy()

            # Convertir la imagen a HSV para facilitar la detección de bordes pese a la iluminación
            plot_marco_img = cv2.cvtColor(plot_marco_img, cv2.COLOR_BGR2HSV)
            
            red_mask, red_mask_thresholded = make_red_mask(plot_marco_img)
            # Mostrar la máscara umbralizada
            #cv2.imshow("Red_mask_thresholded", red_mask_thresholded)
            
            contornos = crear_contornos(red_mask)

            esquinas_detectadas = [contorno.reshape(4, 2) for contorno in contornos][0]


            # Dibujar los contornos en la imagen original (opcional)
            #cv2.drawContours(plot_marco_img, contornos, -1, (0,255,0), 3)
            
            # Convertir la imagen de nuevo a BGR para mostrarla correctamente
            plot_marco_img = cv2.cvtColor(plot_marco_img, cv2.COLOR_HSV2BGR)

            # Nombrar las esquinas del marco exterior e interior
            plot_marco_img = nombrar_esquinas(contornos, plot_marco_img, hwimg)

            # Añadimos al fichero "resultado.txt" las coordenadas del contorno exterior
            contornos_exterior = contornos[0]
            coordenadas = "; ".join([f"{p[0][0]}; {p[0][1]}" for p in contornos_exterior])
            output_file.write(f"{os.path.basename(query_img_path)}; {coordenadas}\n")


            # Mostrar el resultado en pantalla.
            cv2.imshow("Esquinas detectadas", plot_marco_img)
            cv2.waitKey(1000)

            """
            # Calcular el error normalizado entre las esquinas detectadas y las Ground Truth (GT) (opcional)
            if gt_index < len(gt_bloques):
                esquinas_gt = gt_bloques[gt_index]
                error = calcular_error_normalizado(esquinas_detectadas, esquinas_gt)
                print(f"[{os.path.basename(query_img_path)}] Error normalizado: {error:.4f}")
                gt_index += 1
            else:
                print(f"No hay más esquinas GT para comparar.")"""
