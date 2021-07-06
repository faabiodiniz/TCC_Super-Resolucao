import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import sys
from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename
from PIL import Image

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth Ou models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')
# device = torch.device('cuda')

def SaveImage(base, output):
    localImageSave = str(asksaveasfilename(filetypes=[('PNG', '.png')]))
    cv2.imwrite(localImageSave+'.png', output)
    im = Image.open(localImageSave + '.png')
    im.show()

def SelectImage():
    test_img_file = str(askopenfilename())
    ExecSr(test_img_file)

def ExecSr(test_img_file):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    idx = 0
    
    for path in glob.glob(test_img_file):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        #cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
        print("super Resolução completa!")
        SaveImage(base, output)


test_img_file=""

janela = Tk()

lb1 = Label(janela)
lb1.place(x=100, y=100)
janela.title("Tela Inicial")

#Largura x Altura + Dist Esquerda + Dist Topo
janela.geometry("300x300+250+250")

bt = Button(janela, text="Selecione a imagem", command=SelectImage)
bt.place(x=90, y=260)



janela.mainloop()

Tk().withdraw()

