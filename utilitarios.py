import cv2

def leer_db():
    info = open('db.csv', 'r')
    ETIQUETAS = []  
    while (True):  
      registro = info.readline()
      if registro == '':
         break
      ETIQUETAS.append(registro.split(",")[1].rstrip())

    return ETIQUETAS 

etiquetas = leer_db() 

def obtener_etiqueta(identificador):
    etiqueta = ''
    if identificador >= 0:
        etiqueta = "Nombre: " + etiquetas[identificador] 
    else:
        etiqueta = " Desconocido "

    return etiqueta
