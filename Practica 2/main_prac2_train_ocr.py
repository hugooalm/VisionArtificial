# @brief main_text_ocr
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @modified by Hugo Almendro Gil (hugo.almendro.gil@gmail.com)
# @date 2025
# @mod data june 2025


# Imports necesarios para el OCR
import cv2
import os
import pickle
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.linear_model import RANSACRegressor
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ocr_training_data_loader import OCRTrainingDataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA



def process_word_images(test_words_path, classifier, char_size, results_save_path, show_results=False):

    # Crear el directorio de resultados si no existe
    if not os.path.exists(results_save_path):
        os.mkdir(results_save_path)

    results_file_path = os.path.join(results_save_path, "results_text_lines.txt")
    with open(results_file_path, "w", encoding="utf-8") as results_file:
        # Iterar sobre las imágenes en el directorio de prueba
        for root, dirs, files in os.walk(test_words_path):
            for file_name in sorted(files):
                if file_name == 'gt.txt': continue
                file_path = os.path.join(root, file_name)

                # Cargar la imagen en escala de grises, el 0 es lo mismo que si ponemos cv2.IMREAD_GRAYSCALE
                image = cv2.imread(file_path, 0)

                if image is None:
                    print(f"Error al cargar la imagen: {file_path}")
                    continue

                if show_results:
                    # Mostrar la imagen original
                    cv2.imshow('Original Image', image)
                    cv2.waitKey(0)

                # Preprocesar la imagen (binarización)
                Imagen_binaria = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


                # Detectar contornos
                contours, _ = cv2.findContours(Imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                char_rects = []

                # Filtrar contornos para encontrar caracteres
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    # Filtrar caracteres por relación de aspecto y tamaño
                    if 0.2 < aspect_ratio < 1.5 and h > 10:
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

                    # Asegurarse de que el nuevo tamaño es mayor que 10 píxeles
                    new_size = max(w, h)
                    if (original_size < 10) or ():
                        continue
                    
                    # Verificar que el rectángulo no se sale de la imagen
                    if (x < 0) or (y < 0) or (x + w >= image.shape[1]) or (y + h >= image.shape[0]):
                        continue

                    # Centrar el carácter en un cuadrado como en el entrenamiento
                    Icrop = np.zeros((new_size, new_size), dtype=np.uint8)
                    x_0 = int((new_size - w) / 2)
                    y_0 = int((new_size - h) / 2)

                    # Extraer el carácter de la imagen 
                    char_img = Imagen_binaria[y:y + h, x:x + w]
                    Icrop[y_0:y_0 + h, x_0:x_0 + w] = char_img
                    char_img_resized = cv2.resize(Icrop, char_size, interpolation=cv2.INTER_NEAREST)
                    
                    # Conseguir el vector de características del carácter
                    char_vector = char_img_resized.flatten().reshape(1, -1)
                    predicted_char = classifier.predict(char_vector)[0]
                    line_text += predicted_char

                    if show_results:
                        # Mostrar el carácter detectado
                        I2 = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(I2, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
                        cv2.imshow('Letters', I2)
                        cv2.imshow('resize', char_img_resized)
                        cv2.waitKey(0)
                    
                # Guardar resultados en el archivo
                x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0] 
                results_file.write(f"{file_name};{x1};{y1};{x2};{y2};{line_text}\n")
                print(f"Procesado: {file_name}, Texto detectado: {line_text}")


    print(f"Resultados guardados en {results_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crea y ejecuta un detector sobre las imágenes de test')
    parser.add_argument(
        '--train_ocr_path', default="Materiales_Practica2/train_ocr", help='Select the training data dir for OCR')
    parser.add_argument(
        '--test_ocr_char_path', default="Materiales_Practica2/test_ocr_char", help='Imágenes de test para OCR de caracteres')
    parser.add_argument(
        '--test_ocr_words_path', default="Materiales_Practica2/test_ocr_words_plain", help='Imágenes de test para OCR con palabras completas')
    args = parser.parse_args()

    # Configuración de los parámetros del OCR para test de caracteres y palabras

    TEST_OCR_CLASSIFIER_IN_CHARS=True
    TEST_OCR_CLASSIFIER_IN_WORDS=True

    SAVED_OCR_CLF_RandomForest = "RandomForestClassifier.pickle"
    SAVED_OCR_CLF_GaussianNB = "GaussianNB.pickle"
    SAVED_OCR_CLF_SVC = "SVC.pickle"   
    SAVED_OCR_CLF_KNeighbors = "KNeighborsClassifier.pickle"

    OCR_CLF = SAVED_OCR_CLF_RandomForest   # Cambiar para probar con otro clasificador junto descomentar modelo
    
    # Definir clasificadores 

    clf = RandomForestClassifier(n_estimators=100, random_state=42) # Quitar comentario del que se quiera usar
    #clf = GaussianNB()
    #clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    #clf = KNeighborsClassifier(n_neighbors=3)

    n_components = 50
    pca = PCA(n_components=n_components)

    print("Entrenenando OCR...")

    data_ocr = OCRTrainingDataLoader()
    
    if not os.path.exists(OCR_CLF):
        # Cargar los datos de entrenamiento usando OCRTrainingDataLoader
        print("Cargando datos de entrenamiento...")
        
        images_dict = data_ocr.load(args.train_ocr_path)
        print("\n\n\nNúmero de clases:", len(images_dict), "\n\n\n")

        # Convertir las imágenes y etiquetas en listas para el entrenamiento
        X_train = []
        y_train = []
        for label, images in images_dict.items():
            for img in images:
                X_train.append(img.flatten())  # Aplanar la imagen a un vector 1x900
                y_train.append(label)  # La etiqueta es el nombre de la carpeta

        # Convertir las listas a arrays de NumPy
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train)

        
        # Aplicar PCA para reducir la dimensionalidad de las características (opcional y utilizado en SVC sólamente por nosotros)
        #X_train_pca = pca.fit_transform(X_train)

    
        # Entrenar el clasificador RandomForestClassifier
        print("Entrenando el clasificador...")
        clf.fit(X_train, y_train)
        #clf.fit(X_train_pca, y_train) # Utilizar si usamos PCA

        print("Clasificador entrenado.")
        # Guardar el clasificador entrenado
        with open(OCR_CLF, "wb") as pickle_file:
            pickle.dump(clf, pickle_file)
            #pickle.dump({'clf': clf, 'pca': pca}, pickle_file) # Sólo se utiliza para SVC con PCA

    else:
        # Cargar el clasificador entrenado
        with open(OCR_CLF, "rb") as pickle_file:
            clf = pickle.load(pickle_file)
            #data = pickle.load(pickle_file) # Sólo se utilizan para SVC con PCA
            #clf = data['clf']
            #pca = data['pca']
            print("Cargado clasificador desde el archivo pickle.")

    if TEST_OCR_CLASSIFIER_IN_CHARS:
        # Cargar las imágenes de prueba para el OCR de caracteres
        print("Cargando datos de prueba ...")
        test_dict_chars = data_ocr.load(args.test_ocr_char_path)

        # Convertir el diccionario en arrays de características y etiquetas
        X_test_chars = []
        y_test_chars = []
        for label, images in test_dict_chars.items():
            for img in images:
                X_test_chars.append(img.flatten())  # Aplanar la imagen a un vector 1x900
                y_test_chars.append(label)  # La etiqueta es el nombre de la carpeta

        # Convertir las listas a arrays de NumPy
        X_test_chars = np.array(X_test_chars, dtype=np.float32)
        y_test_chars = np.array(y_test_chars)
        
        # Aplicar PCA para reducir la dimensionalidad de las características (opcional y utilizado en SVC sólamente por nosotros)
        #X_test_chars_pca = pca.transform(X_test_chars)


        print("Ejecutando clasificador en conjunto de test ...")
        estimated_test_chars = clf.predict(X_test_chars)
        #estimated_test_chars = clf.predict(X_test_chars_pca) # Utilizar si usamos PCA
        estimated_test_chars = [pred.upper() for pred in estimated_test_chars]

        # Mostrar resultados del OCR de caracteres
        
        accuracy_chars = accuracy_score(y_test_chars, estimated_test_chars)
        print("\n\n    Accuracy char OCR = ", accuracy_chars)
        f1_macro = f1_score(y_test_chars, estimated_test_chars, average='macro')
        print("    F1-macro char OCR = ", accuracy_chars)
        precision_macro = precision_score(y_test_chars, estimated_test_chars, average='macro')
        print("    Precision-macro char OCR = ", precision_macro)
        recall_macro = recall_score(y_test_chars, estimated_test_chars, average='macro')
        print("    Recall-macro char OCR = ", recall_macro)
        
        # Reporte completo por clase (incluyendo precisión, recall y F1-score tanto macro como weighted)
        print("\nClassification report:\n", classification_report(y_test_chars, estimated_test_chars))

        # Matriz de confusión
        cm = confusion_matrix(y_test_chars, estimated_test_chars)
        
        # Visualización gráfica
        fig, ax = plt.subplots(figsize=(12, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_chars))
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        # Quitar los números de la matriz de confusión
        for texts in disp.text_.ravel():
            texts.set_visible(False)
        plt.title("Matriz de Confusión OCR")
        plt.show()


    if TEST_OCR_CLASSIFIER_IN_WORDS:
        # Cargar las imágenes de prueba para el OCR de palabras
        print("Cargando y procesando palabras de prueba ...")
        
        # Crear el directorio de resultados si no existe
        results_save_path = "results_ocr_words_plain"
        try:
            os.mkdir(results_save_path)
        except:
            print('El directorio ya esta creado  "' + results_save_path + '"')

        # Ejecutar el clasificador en las imágenes de prueba que contienen palabras
        process_word_images(args.test_ocr_words_path, clf, (30, 30), "results_ocr_words_plain")

