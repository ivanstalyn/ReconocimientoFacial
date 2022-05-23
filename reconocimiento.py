#!/usr/bin/python3
'''
 Este script sirve para utilizar utilizar el modelo entrenado de imagenes sobre una cámara de video o fotografia
 '''
import cv2, argparse
from utilitarios import obtener_etiqueta

def reconocer(opciones):
 
  face_classif = cv2.CascadeClassifier('./HAAR/haarcascade_frontalface_default.xml')
  reconocimiento = cv2.face.LBPHFaceRecognizer_create()#2, 2, 7, 7, 15)
  reconocimiento.read('modeloLBPHFace.xml')

  camara = cv2.VideoCapture(1)

  while True:
    ret, imagen_original = camara.read()
    imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)  
    rostros = face_classif.detectMultiScale(imagen_gris, 1.3, 5)  

    for (x, y, w, h) in rostros:  
      cara_gris = cv2.resize((imagen_gris[y: y + h, x: x + w]), (150, 150),interpolation= cv2.INTER_CUBIC)
      identificador, conf = reconocimiento.predict(cara_gris)  
      etiqueta = obtener_etiqueta(int(identificador))
      cv2.putText(imagen_original,etiqueta,(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
      cv2.rectangle(imagen_original, (x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Reconociendo imagen',imagen_original)  
          
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