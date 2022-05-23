#!/usr/bin/python3
'''
 Este script sirve para utilizar utilizar el modelo entrenado de imagenes sobre una cámara de video o fotografia
 '''
import cv2, argparse
from utilitarios import obtener_etiqueta

def reconocer(opciones):
 
  face_classif = cv2.CascadeClassifier('./HAAR/haarcascade_frontalface_default.xml')
  
  recon_eigen = cv2.face.EigenFaceRecognizer_create() # (15, 4000) 
  recon_eigen.read('modeloEigenFace.xml')

  recon_fisher = cv2.face.FisherFaceRecognizer_create() #(2, 40)  
  recon_fisher.read('modeloFisherFace.xml')

  recon_lbph = cv2.face.LBPHFaceRecognizer_create()#(2, 2, 7, 7, 15)  
  recon_lbph.read('modeloLBPHFace.xml')


  camara = cv2.VideoCapture(1)

  while True:
    ret, imagen_original = camara.read()
    imagen_original_eigen = imagen_original.copy()
    imagen_original_fisher = imagen_original.copy()
    imagen_original_lbph = imagen_original.copy()

    imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)  
    rostros = face_classif.detectMultiScale(imagen_gris, 1.3, 5)  

    for (x, y, w, h) in rostros:  
      cara_gris = cv2.resize((imagen_gris[y: y + h, x: x + w]), (150, 150),interpolation= cv2.INTER_CUBIC)
      identificador_eigen, conf_eigen = recon_eigen.predict(cara_gris)
      identificador_fisher, conf_fisher = recon_fisher.predict(cara_gris)
      identificador_lbph, conf_lbph = recon_lbph.predict(cara_gris)

      etiqueta_eigen = obtener_etiqueta(int(identificador_eigen))
      etiqueta_fisher = obtener_etiqueta(int(identificador_fisher))
      etiqueta_lbph = obtener_etiqueta(int(identificador_lbph))

      print(f'{identificador_eigen},{etiqueta_eigen},{conf_eigen},{identificador_fisher},{etiqueta_fisher},{conf_fisher},{identificador_lbph},{etiqueta_lbph},{conf_lbph}')

      cv2.putText(imagen_original_eigen,etiqueta_eigen,(x,y-25),2,1.1,(255,0,0),1,cv2.LINE_AA)
      cv2.putText(imagen_original_fisher,etiqueta_fisher,(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
      cv2.putText(imagen_original_lbph,etiqueta_lbph,(x,y-25),2,1.1,(0,0,255),1,cv2.LINE_AA)
      cv2.rectangle(imagen_original_eigen, (x,y),(x+w,y+h),(255,0,0),2)
      cv2.rectangle(imagen_original_fisher, (x,y),(x+w,y+h),(0,255,0),2)
      cv2.rectangle(imagen_original_lbph, (x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('Reconociendo imagen EIGEN',imagen_original_eigen)
    cv2.imshow('Reconociendo imagen FISHER',imagen_original_fisher)
    cv2.imshow('Reconociendo imagen LBPH',imagen_original_lbph)  
          
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit if the key is Q
        break
  
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("-c", 
                      "--camara", 
                      help="DEVICE_ID de la camara a utilizar. ",
                      default='0')
  
  parser.add_argument("-m", 
                      "--modelo", 
                      help="Tipo de modelo",
                      choices=['eigen','fisher','lbph'],
                      default='lbph')

  parser.add_argument("-d", 
                      "--directorio", 
                      help="Directorio de imagenes",
                      default='imagenes')
               

  args = parser.parse_args()
  reconocer(args)
  
if __name__ == "__main__":
  main()
  print(f'Finalizado.')