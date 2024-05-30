################################################################################
# DustDetection 
# helpers.py
# Xavier Zientarski
# 2024-05-29
################################################################################
import numpy as np
from PIL import Image, ImageOps
import torch
import cv2

################################################################################
#                                                         
################################################################################
def redimensionar_image(im):
    '''
    Redimensiona la imagen a 1024x1024
    '''
    ori_frame = Image.open(im)
    newsize = (1024, 1024)
    print("\tRedimensión")
    return ori_frame.resize(newsize)
################################################################################
#                                                         
################################################################################
def identificar_polvo(ori_frame, model, img_transform):

    dust_array = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (H, W, _) = np.shape(ori_frame)
    frame = img_transform(ori_frame)
    input = frame.unsqueeze(0).float()
    input = input.to(device)
    mask = torch.sigmoid(model(input)['out'])
    mask = mask.squeeze(0).squeeze(0)

    #Dust Probability clip
    #mask = mask > 0.99
    mask = mask > 0.05

    mask = mask.detach().cpu().numpy()
    mask = mask.astype(np.float32)
    mask = cv2.resize(mask, (W, H))
    mask = np.expand_dims(mask, axis=-1)

    dust_array.append(np.sum(mask))

    invmask = mask.copy()
    invmask = abs(mask - 1)
    combine_frame = np.asanyarray(ori_frame).copy()

    combine_frame[:,:,1] = combine_frame[:,:,1] * invmask[:,:,0]
    combine_frame = combine_frame.astype(np.uint8)
    print("\tIdentificación de polvo")
    return Image.fromarray(combine_frame.astype('uint8')), mask
################################################################################
#                                                         
################################################################################
def cuadro_porcentajes(mask):
  def cantidadPolvo(im):
    total = np.prod(im.shape)
    # print("Total", total)
    polvo = im.sum()
    # print("Polvo",polvo)
    porcentaje = round((polvo / total) * 100, 2)
    #print(f"Porcentaje de polvo en la celda: {porcentaje}")
    return porcentaje
  #-----------------------------------------------------------------------------
  incremento = 8
  print("\tCuadro de porcentajes")

  valor = int(1024 / incremento)

  # Valor de y
  porcentajes = []
  y_start = 0
  for y in range(1, incremento + 1):
    #y_start = 0
    y_end = y * valor
    #print(y_end)

    x_start = 0
    for x in range(1, incremento + 1):
      # x_start =
      x_end = x * valor

      # Cálculo
      porcentaje_polvo = cantidadPolvo(mask[y_start:y_end, x_start:x_end])
      #print("_Y_:", f"{y_start} x {y_end}", "\t\t_X_:", f"{x_start} x {x_end}", "Porcentaje:", porcentaje_polvo)
      porcentajes.append(porcentaje_polvo)
      # TO DO cálculo

      x_start = x_end

    # Actualizar el limite
    y_start = y_end
  return np.reshape(porcentajes, (-1, 8))
################################################################################
#                                                         
################################################################################
def colorear_imagen(img, cuadro_porcentajes, cuadriculado=False):
  '''
  Regresa una imagen completamente coloreada acorde al nivel de polvo
  '''
  #-----------------------------------------------------------------------------
  def cuadricular(img):
    img1 = np.array(img.convert('RGB'))
    img1

    # Definir el valor para la cantidad de cuadros -> x * x
    incremento = 8
    print("\tCuadriculado")

    valor = int(1024 / incremento)

    for i in range(1, incremento):
      for ii in range(1024):
        #print(i * 128)
        img1[ii, i * valor] = [0,0,0]
        img1[i * valor, ii] = [0,0,0]

    return Image.fromarray(np.uint8(img1)).convert('RGB')
  #-----------------------------------------------------------------------------
  def tint_image(src, color="#FFFFFF"):
    src.load()
    r, g, b, alpha = src.split()
    gray = ImageOps.grayscale(src)
    result = ImageOps.colorize(gray, (0, 0, 0, 0), color)
    result.putalpha(alpha)
    return result
  #-----------------------------------------------------------------------------
  def coloreado(cuadro_porcentajes, VERDE, AMARILLO, ROJO):
    incremento = 8
    print("\tColoreado")

    valor = int(1024 / incremento)
    prev_y = 0
    prev_x = 0
    for i in range(1, incremento + 1): # Y
      lim_y = i * valor
      for ii in range(1, incremento + 1): # X
        lim_x = ii * valor
        act = cuadro_porcentajes[i - 1][ii - 1]
        if act > 15: # ROJO
          VERDE[prev_y:lim_y, prev_x:lim_x] =     ROJO[prev_y:lim_y, prev_x:lim_x]
        elif act > 5: # AMARRILLO
          VERDE[prev_y:lim_y, prev_x:lim_x] = AMARILLO[prev_y:lim_y, prev_x:lim_x]

        prev_x = lim_x

      prev_x = 0
      prev_y = lim_y

    return VERDE
  #-----------------------------------------------------------------------------
  # Acciones
  if cuadriculado:
    img = cuadricular(img)
  img = img.convert("RGBA")
  VERDE    = np.array(tint_image(img, "#A8E7B3").convert( 'RGB' ))
  AMARILLO = np.array(tint_image(img, "#EEE22C").convert( 'RGB' ))
  ROJO     = np.array(tint_image(img, "#EE3A2C").convert( 'RGB' ))

  return Image.fromarray(coloreado(cuadro_porcentajes, VERDE, AMARILLO, ROJO))
################################################################################
################################################################################