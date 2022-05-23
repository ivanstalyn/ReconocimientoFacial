#!/usr/bin/python3
'''
 Este script captura imagenes de rostros dentro de 
 videos MP4.
 
 El etiquetado de los rostros se realiza con base en el nombre del archivo mp4.
 
 Es así que se recomienda usar nombres de personas en el nombre de los archivos. Por ejemplo, 
 Ivan-Muela.mp4 o lo que se ajuste a tus necesidades de etiquetado
'''
import cv2,  glob, os, argparse, imutils

def capturainfo(opciones):
  
  
  for filename in glob.glob(opciones.videos):

    archivo = filename.split('/')
    archivo = archivo[len(archivo)-1]
    etiqueta = archivo.split('.')[0]
    print(f'Archivo: {archivo}')
    print(f'Etiqueta: {etiqueta}')
    
    ruta_fotos = opciones.fotos + '/' + etiqueta

    if(not os.path.exists(ruta_fotos)):
      print(f'Carpeta creada: {ruta_fotos}')
      os.makedirs(ruta_fotos)
      
    try:
      captura = cv2.VideoCapture(filename)
    except Exception as error:
      print('Error con algo de la captura del video: ' + str(error))
    
    counter = 0
    face_classif = cv2.CascadeClassifier('./HAAR/haarcascade_frontalface_default.xml')
    while True:
      ret, img_color = captura.read()
      
      if ret == False: break
      
      img_color =  imutils.resize(img_color, width=640)
      img_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
      faces = face_classif.detectMultiScale(img_gris,1.3,5)
      
      for (x,y,w,h) in faces:
          cv2.rectangle(img_color, (x,y),(x+w,y+h),(0,255,0),2)
          rostro = img_gris[y:y+h,x:x+w]
          rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
          cv2.imwrite(f'{ruta_fotos}/imagen_{counter}.jpg',rostro)
          counter = counter + 1
          
      cv2.imshow('Reconociendo imagen',img_color)  

      k =  cv2.waitKey(1)
      if k == 27 or counter >= int(opciones.muestras):
        break
    
    captura.release()
    cv2.destroyAllWindows()



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", 
                      "--videos", 
                      help="Ruta al archivo MP4",
                      default='videos/*')
  parser.add_argument("-f", 
                      "--fotos", 
                      help="Directorio de destino de las fotos de rostros que se extrajeron del MP4.",
                      default='fotos')

  parser.add_argument("-m", 
                      "--muestras", 
                      help="Número de muestras por video MP4.",
                      default='300')                  

  args = parser.parse_args()
  capturainfo(args)

if __name__ == "__main__":
    main()
    print(f'Finalizado.')