################################################################################
# DustDetection 
# model.py
# Xavier Zientarski
# 2024-05-29
################################################################################
import helpers as hp
import os
import torch
from torchvision.transforms import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
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
directory = os.fsencode('images/')
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    nombre, _ = filename.split('.')

    original = hp.redimensionar_image("images/" + filename)
    original.save('output/original/' + nombre + '_original.png')

    polvo, mask = hp.identificar_polvo(original, model, img_transform)
    polvo.save('output/polvo/' + nombre + '_polvo.png')

    cuadro_porcentajes = hp.cuadro_porcentajes(mask)
    ajedrez = hp.colorear_imagen(original, cuadro_porcentajes, cuadriculado=True)
    ajedrez.save('output/ajedrez/' + nombre + '_ajedrez.png')

    hp.json_report(nombre, cuadro_porcentajes)
################################################################################
################################################################################