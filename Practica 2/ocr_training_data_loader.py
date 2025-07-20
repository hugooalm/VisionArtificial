# @brief Tracker
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025x
#

import numpy as np
import cv2
import os


class OCRTrainingDataLoader:
    """
    Class to read cropped images from the OCR training data generated on folders (one for each char).
    """

    def __init__(self, char_size=(30,30)):
        self.name = 'URJC-OCR-TRAIN'
        self.char_size = char_size

    def load(self, data_path):
        """
         Given a directory where dataset is, read all the images on each folder (= char).

         :return images where images is a dictionary of lists of images where the key is the class (= char).
        """
        images = dict()

        for root, dirs, files in os.walk(data_path):
            if len(dirs) == 0:
                class_label = os.path.basename(root)
                # Solo añadir si es una clase válida (un solo carácter alfanumérico)
                if len(class_label) == 1 and class_label.isalnum():
                    print(f"====> Loading images from: {class_label}")
                    images[class_label] = self.__load_images(root, '', self.char_size)
            for name in sorted(dirs):
                subfolder_path = os.path.join(root, name)
                if name == "may" or name == "min":
                    for subdir in sorted(os.listdir(subfolder_path)):
                        subdir_path = os.path.join(subfolder_path, subdir)
                        if os.path.isdir(subdir_path):
                            # Solo añadir si es una clase válida
                            if len(subdir) == 1 and subdir.isalnum():
                                print(f"====> Loading images from: {subdir}")
                                images[subdir] = self.__load_images(subfolder_path, subdir, self.char_size)
                else:
                    # Solo añadir si es una clase válida
                    if len(name) == 1 and name.isalnum():
                        print(f"====> Loading images from: {name}")
                        images[name] = self.__load_images(root, name, self.char_size)

        return images

    def __load_images(self, data_path, char_data_dir, chars_size, show_results=False):

        images = []
        for i, name in enumerate(sorted(os.listdir(os.path.join(data_path, char_data_dir)))):
            if name == 'gt.txt': continue

            I = cv2.imread(os.path.join(data_path, char_data_dir, name), 0)
 
            if not type(I) is np.ndarray:  # file it is not an image.
                print("*** ERROR: Couldn't read image " + name)

                continue

            Imagen_binaria = I.copy()
            Imagen_binaria = cv2.adaptiveThreshold(I, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            contours, _ = cv2.findContours(Imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                r = cv2.boundingRect(contour)

                x, y, w, h = r
                original_size = max(w, h)

                if (w==1) or (h==1):
                    continue

                r = (x, y, w, h)
                # Aumentar el tamaño del rectángulo para centrar el carácter
                p = 0.1
                #x = max(int(round(r[0] - r[2] * p)), 0)
                y = max(int(round(r[1] - r[3] * p)), 0)
                #w = int(round(r[2] * (1. + 2.0 * p)))
                h = int(round(r[3] * (1. + 2.0 * p)))

                new_size = max(w, h)
                
                # Asegurarse de que el nuevo tamaño es mayor que 10 píxeles
                if (original_size < 10) or ():
                    continue

                # Verificar que el rectángulo no se sale de la imagen
                if (x < 0) or (y < 0) or (x + w >= I.shape[1]) or (y + h >= I.shape[0]):
                    continue

                # Centrar el carácter en un cuadrado 
                Icrop = np.zeros((new_size, new_size), dtype=np.uint8)
                x_0 = int((new_size - w) / 2)
                y_0 = int((new_size - h) / 2)

                # Extraer el carácter de la imagen 
                Icrop[y_0:y_0 + h, x_0:x_0 + w] = Imagen_binaria[y:y + h, x:x + w]
                Iresize = cv2.resize(Icrop, chars_size, interpolation=cv2.INTER_NEAREST)
                images.append(Iresize)


                # Plot results
                if show_results and (i==1):
                    
                    I2 = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
                    x, y, w, h = r
                    cv2.rectangle(I2, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
                    cv2.imshow('Letters', I2)
                    cv2.imshow('crop', cv2.resize(Icrop, None, fx=4.0, fy=4.0))
                    cv2.imshow('resize', cv2.resize(Iresize, None, fx=4.0, fy=4.0))
                    cv2.waitKey(1000)
            
        return images

    def show_image_examples(self, images, num_imgs_per_class=5):
        Iexamples = None
        for key in images:
            examples = [img for i, img in enumerate(images[key]) if (i < num_imgs_per_class)]

            Irow = None
            num_imgs = 0
            for e in examples:
                if Irow is None:
                    Irow = e
                else:
                    Irow = np.hstack((Irow, e))
                num_imgs += 1

                if num_imgs == num_imgs_per_class:
                    break

            if Iexamples is None:
                Iexamples = Irow
            else:
                Iexamples = np.vstack((Iexamples, Irow))

        return Iexamples
