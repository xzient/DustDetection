################################################################################
# DustDetection 
# model.py
# Xavier Zientarski
# 2024-05-29
################################################################################
import helpers as hp
import get_images as gi
import os
import torch
from pathlib import Path
from torchvision.transforms import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
################################################################################
#                                                         
################################################################################
# Get images
dd, hh, mm = gi.get_today_str()
sub_directory = dd + '-' + str(hh).zfill(2) + '-' +str(mm).zfill(2) + '/'

print(sub_directory)

gi.get_images(sub_directory)
################################################################################
#                                                         
################################################################################
model = deeplabv3_mobilenet_v3_large(weights=None, progress=True, num_classes=1, aux_loss=None)
model_name = 'DeepLabV3-MobileNet-V3-Large'
dataset_name = 'URDE_dataset_897'
direccion_modelo = "src/Pretrained/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load(direccion_modelo + 'DeepLabV3-MobileNet-V3-Large_dataset_897.pth', map_location=torch.device(device)))
model.eval()
model.to(device)#cuda()

totalframecount = 1
idx = 0
#font = cv2.FONT_HERSHEY_SIMPLEX
img_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
################################################################################
#                                                         
################################################################################
directory = os.fsencode('images/' + sub_directory)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    nombre, _ = filename.split('.')

    original = hp.redimensionar_image("images/"  + sub_directory + filename)

    Path( 'output/original/' + sub_directory).mkdir(parents=True, exist_ok=True)
    original.save('output/original/'  + sub_directory + nombre + '_original.png')

    polvo, mask = hp.identificar_polvo(original, model, img_transform)
    Path( 'output/polvo/' + sub_directory).mkdir(parents=True, exist_ok=True)
    polvo.save('output/polvo/'  + sub_directory + nombre + '_polvo.png')

    cuadro_porcentajes = hp.cuadro_porcentajes(mask)
    ajedrez = hp.colorear_imagen(original, cuadro_porcentajes, cuadriculado=True)
    Path( 'output/ajedrez/' + sub_directory).mkdir(parents=True, exist_ok=True)
    ajedrez.save('output/ajedrez/' + sub_directory  + nombre + '_ajedrez.png')

    Path( 'output/report/' + sub_directory).mkdir(parents=True, exist_ok=True)
    hp.json_report(nombre, cuadro_porcentajes, sub_directory)
################################################################################
################################################################################