#!/usr/bin/python3
'''
 Este script genera el modelo entrenado de imagenes de rostros sobre la base de los archivos generados con capturaMP4.py
 '''
import cv2, os, argparse, numpy as np

def entrenar(opciones):
 
  db_contenido = ''  
  rutas_imagenes = [os.path.join(opciones.fotos, f) for f in os.listdir(opciones.fotos)]
  lista_rostros = []
  etiquetas = []
  
  #Obtener las imagenes y datos.
  etiqueta = 0
  for ruta in rutas_imagenes:
    db_contenido = f'{db_contenido}{etiqueta},{ruta.replace(opciones.fotos + "/", "")}\n'
    for archivo in os.listdir(ruta):
      imagen = cv2.imread(f'{ruta}/{archivo}',0)
      etiquetas.append(etiqueta)
      lista_rostros.append(imagen)
    etiqueta = etiqueta + 1
    
  
  #Se guarda las etiquetas y ids
  info = open('db.csv', "w")
  info.write(db_contenido)
  info.close()

  #Entrenamiento
  e = opciones.entrenamiento

  if(e == 'eigen' or e == 'todo'):
    print("EIGEN")
    eigen_face = cv2.face.EigenFaceRecognizer_create(15)  
    eigen_face.train(lista_rostros, np.array(etiquetas))
    eigen_face.write('modeloEigenFace.xml')

  if(e == 'fisher' or e == 'todo'):
    print("FISHER")
    fisher_face = cv2.face.FisherFaceRecognizer_create(12)  
    fisher_face.train(lista_rostros, np.array(etiquetas))
    fisher_face.write('modeloFisherFace.xml')
  
  if(e == 'lbph' or e == 'todo'):
    print("LBPH")
    lbph_face = cv2.face.LBPHFaceRecognizer_create(1, 1, 7, 7) 
    lbph_face.train(lista_rostros, np.array(etiquetas))
    lbph_face.write('modeloLBPHFace.xml')


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("-f", 
                      "--fotos", 
                      help="Directorio de destino de las fotos de rostros que se extrajeron del MP4.",
                      default='fotos')
  
  parser.add_argument("-e", 
                      "--entrenamiento", 
                      help="Tipo de entrenamiento",
                      choices=['eigen','fisher','lbph','todo'],
                      default='lbph')
               

  args = parser.parse_args()
  entrenar(args)
  
if __name__ == "__main__":
  main()
  print(f'Finalizado.')