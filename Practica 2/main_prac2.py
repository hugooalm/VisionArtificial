
import cv2
import numpy as np
import os
import pickle
from sklearn.linear_model import RANSACRegressor
import Levenshtein
from evaluar_resultados_test_ocr import levenshtein_distance
from model3d import Model3D
from glob import glob

def load_all_3d_models(models_dir):
    models_dict = {}
    for file in os.listdir(models_dir):
        if file.lower().endswith('.obj'):
            key = os.path.splitext(file)[0].upper()
            model = Model3D()
            model.load_from_obj(os.path.join(models_dir, file))
            models_dict[key] = model
    return models_dict

def predict_word(image, classifier, char_size):
    # Cargar la imagen en escala de grises
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocesar la imagen (binarización)
    #Imagen_binaria = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    _, Imagen_binaria = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #kernel = np.ones((2,2), np.uint8)
    #Imagen_binaria = cv2.morphologyEx(Imagen_binaria, cv2.MORPH_CLOSE, kernel) # Para cerrar huecos en el caracter
    #Imagen_binaria = cv2.morphologyEx(Imagen_binaria, cv2.MORPH_DILATE, kernel) # Para agrandar el caracter
   

    # Detectar contornos
    contours, _ = cv2.findContours(Imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_rects = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = w * h
        max_area = 0.05 * image.shape[0] * image.shape[1]  # Filtramos aquellos cuyo area es mayor al 5% de la imagen para evitar ruido

        # Filtrar caracteres por relación de aspecto y tamaño
        if 0.1 < aspect_ratio < 4 and h > 4 and area < max_area:
            char_rects.append((x, y, w, h))

    # Ordenar los rectángulos por su posición horizontal
    char_rects = sorted(char_rects, key=lambda r: r[0])

    # Encontrar caracteres alineados usando RANSAC
    centers = np.array([[x + w / 2, y + h / 2] for x, y, w, h in char_rects])
    if len(centers) > 1:
        ransac = RANSACRegressor()
        ransac.fit(centers[:, 0].reshape(-1, 1), centers[:, 1])
        inliers = ransac.inlier_mask_
        char_rects = [char_rects[i] for i in range(len(char_rects)) if inliers[i]]

     # Crear una copia de la imagen original para dibujar
    img_draw = image.copy()

    # Dibujar los rectángulos de los caracteres detectados
    for x, y, w, h in char_rects:
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Dibujar la línea que une los centros de los caracteres
    if len(char_rects) > 1:
        pts = np.array([[int(x + w / 2), int(y + h / 2)] for x, y, w, h in char_rects], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_draw, [pts], False, (255, 0, 0), 2)

    # Mostrar la imagen con las detecciones
    cv2.imshow('Deteccion de caracteres y linea', img_draw)
    
    
    # Clasificar caracteres y generar el string de salida
    line_text = ""
    for r in char_rects:

        x, y, w, h = r
        original_size = max(w, h)

        if (w==1) or (h==1):
            continue

        # Aumentar el tamaño del rectángulo para centrar el carácter
        p = 0.1
        #x = max(int(round(r[0] - r[2] * p)), 0)
        y = max(int(round(r[1] - r[3] * p)), 0)
        #w = int(round(r[2] * (1. + 2.0 * p)))
        h = int(round(r[3] * (1. + 2.0 * p)))

        new_size = max(w, h)
        if (original_size < 10) or ():
            continue

        if (x < 0) or (y < 0) or (x + w >= image.shape[1]) or (y + h >= image.shape[0]):
            continue

        Icrop = np.zeros((new_size, new_size), dtype=np.uint8)
        x_0 = int((new_size - w) / 2)
        y_0 = int((new_size - h) / 2)

        char_img = Imagen_binaria[y:y + h, x:x + w]
        Icrop[y_0:y_0 + h, x_0:x_0 + w] = char_img
        char_img_resized = cv2.resize(Icrop, char_size, interpolation=cv2.INTER_NEAREST)
        
        char_vector = char_img_resized.flatten().reshape(1, -1)
        predicted_char = classifier.predict(char_vector)[0]
        line_text += predicted_char

    return line_text

def find_best_model(word, models):
    min_dist = float('inf')
    best_key = None
    for key in models.keys():
        dist = levenshtein_distance(word, key)
        if dist < min_dist:
            min_dist = dist
            best_key = key
    return best_key

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

def ejercicio2_1_2(query_img, pts, htimg, dst):
            
    img_temp = query_img.copy()
    hwt = cv2.getPerspectiveTransform(pts_mm, pts)
    img = cv2.polylines(img_temp, [np.int32(dst)],True,255,3, cv2.LINE_AA)
    hwimg = htimg @ hwt
    img = cv2.resize(img, (1400, 900))
    return hwt, hwimg, img

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

def ejercicio2_1_4(query_img, model, P, model_3d_file):
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
    
    return plot_img


########################### Ejercicio 4 ###########################

# Cargar el clasificador OCR
SAVED_OCR_CLF_RandomForest = "RandomForestClassifier.pickle"
SAVED_OCR_CLF_GaussianNB = "GaussianNB.pickle"
OCR_CLF = SAVED_OCR_CLF_RandomForest

if not os.path.exists(OCR_CLF):
    print("No se ha encontrado el clasificador seleccionado, prueba otro.")
    exit(1)
else:
    with open(OCR_CLF, "rb") as pickle_file:
            clf = pickle.load(pickle_file)
            print("Cargado clasificador desde el archivo pickle.")

# Cargar los modelos 3D
models_path = 'Materiales_Practica2/3d_models'
models = load_all_3d_models(models_path)

# Cargar la imagen de la plantilla escaneada    
template_img_path = "Materiales_Practica2/test_template_ocr_simple/template_cropped.png"
template_img = cv2.imread(template_img_path)

if template_img is None:
    print("No puedo encontrar la imagen " + template_img_path)

# Leer la matriz de intrínsecos de la cámara.
K = np.loadtxt("Materiales_Practica2/test_template_ocr_simple/intrinsics.txt")

# Crear el detector SIFT 
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(template_img,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

# Obtener las dimensiones de la plantilla
h,w,color = template_img.shape

# Definir los puntos de la plantilla tanto en píxeles como en milímetros
pts = np.float32([ [0,0],[w-1,0],[w-1,h-1],[0,h-1] ]).reshape(-1,1,2)
pts_mm = np.float32([ [0,0],[0,210],[185,210],[185,0] ]).reshape(-1,1,2)

# Crear el matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

paths = sorted(glob(os.path.join("Materiales_Practica2/test_template_ocr", "*.jpg")))

resultados_path = "resultados_modelos3d.txt"
with open(resultados_path, "w", encoding="utf-8") as f:
    pass  # Esto vacía el archivo

for f in paths:
    query_img_path = f
            
    query_img = cv2.imread(query_img_path)
    if query_img is None:
        print("No puedo encontrar la imagen " + query_img_path)
        continue

    ######################### Parte de la práctica 1 #########################

    img_matches, htimg, matchesMask, dst =  ejercicio2_1_1(query_img, pts)
    hwt, hwimg, img_homografia =  ejercicio2_1_2(query_img, pts, htimg, dst)
    P = ejercicio2_1_3(K, hwimg)

    ######################### Parte de la práctica 1 #########################

    # Cogemos los 4 puntos de la zona de la derecha de la homografía calculada, ya que 
    # ahí se encuentra el resto del folio, donde está la palabra.
    pto_arriba_der_folio = np.float32([[[w+580, 0]]]) 
    pto_abajo_der_folio = np.float32([[[w+580, h]]]) 
    pto_abajo_izq_folio = np.float32([[[w + 10, h]]]) # w + 10 para que no se solape con el borde rojo
    pto_arriba_izq_folio = np.float32([[[w + 10, 0]]])

    # Transformar a la imagen principal usando la homografía
    pto_arriba_der_folio_i = cv2.perspectiveTransform(pto_arriba_der_folio, htimg)
    pto_abajo_der_folio_i = cv2.perspectiveTransform(pto_abajo_der_folio, htimg)
    pto_abajo_izq_folio_i = cv2.perspectiveTransform(pto_abajo_izq_folio, htimg)
    pto_arriba_izq_folio_i = cv2.perspectiveTransform(pto_arriba_izq_folio, htimg)


    x1, y1 = int(pto_arriba_der_folio_i[0, 0, 0]), int(pto_arriba_der_folio_i[0, 0, 1]) # arriba derecha
    x2, y2 = int(pto_abajo_der_folio_i[0, 0, 0]), int(pto_abajo_der_folio_i[0, 0, 1]) # abajo derecha
    x3, y3 = int(pto_abajo_izq_folio_i[0, 0, 0]), int(pto_abajo_izq_folio_i[0, 0, 1]) # abajo izquierda
    x4, y4 = int(pto_arriba_izq_folio_i[0, 0, 0]), int(pto_arriba_izq_folio_i[0, 0, 1]) # arriba izquierda

    # Dibujar el punto sobre la imagen principal
    img_con_punto = query_img.copy()
    
    cv2.circle(img_con_punto, (x1, y1), 10, (0, 0, 255), -1)  # arriba derecha
    cv2.circle(img_con_punto, (x2, y2), 10, (0, 0, 255), -1)  # abajo derecha
    cv2.circle(img_con_punto, (x3, y3), 10, (0, 0, 255), -1)  # abajo izquierda
    cv2.circle(img_con_punto, (x4, y4), 10, (0, 0, 255), -1)  # arriba izquierda

    # Crear un array con los puntos del polígono
    pts_poly = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
    pts_poly = pts_poly.reshape((-1, 1, 2))

    # Crear una máscara del mismo tamaño que la imagen
    mask = np.zeros(query_img.shape[:2], dtype=np.uint8)

    # Rellenar el polígono en la máscara
    cv2.fillPoly(mask, [pts_poly], 255)

    # Aplicar la máscara a la imagen de la escena
    zona = cv2.bitwise_and(query_img, query_img, mask=mask)

    # Recortar la zona delimitada por los puntos
    x_min = np.min([x1, x2, x3, x4])
    x_max = np.max([x1, x2, x3, x4])
    y_min = np.min([y1, y2, y3, y4])
    y_max = np.max([y1, y2, y3, y4])
    zona_recortada = zona[y_min:y_max, x_min:x_max]

    # Puntos de la zona en la imagen de la escena 
    pts_src = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # Definimos un tamaño que se ajuste al tamaño que queremos para la zona rectificada sin perspectiva
    width = 250
    height = 500 

    # Puntos destino (rectángulo rectificado)
    pts_dst = np.float32([[width-1, 0], [width-1, height-1], [0, height-1], [0, 0]])

    # Calcula la homografía
    H = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Aplica la transformación para obtener la vista "cenital"
    zona_rectificada = cv2.warpPerspective(query_img, H, (width, height))

    # Mostrar la zona rectificada (cenital)
    cv2.imshow("Zona rectificada (cenital)", zona_rectificada)

    # Pasamos esa vista zenital al OCR para detectar la palabra
    word = predict_word(zona_rectificada, clf, (30, 30))
    
    # Convertir la palabra a mayúsculas
    word = word.upper()
    print("Palabra detectada:", word)

    # Después de obtener la palabra reconocida:
    best_model_key = find_best_model(word, models)
    print(f"Modelo 3D más parecido: {best_model_key}")
    best_model = models[best_model_key]

    
    # Guardar resultados en un archivo txt
    with open(resultados_path, "a", encoding="utf-8") as f_out:
        nombre_archivo = os.path.basename(query_img_path)
        f_out.write(f"{nombre_archivo};-1;-1;-1;-1;{word}\n")


    # Proyectar el modelo correspondiente sobre la plantilla de la escena
    plot_img = ejercicio2_1_4(query_img, best_model, P, os.path.join(models_path, best_model_key + ".obj"))
    cv2.imshow("3D info on images", cv2.resize(plot_img, None, fx=1, fy=1))
    cv2.waitKey(0) # Presinar cualquier tecla para continuar
    cv2.destroyAllWindows()
    

