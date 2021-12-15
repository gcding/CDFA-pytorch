import torch
import torchvision.transforms as standard_transforms
from torch.autograd import Variable

from networks.CDFA import CDFA

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import getopt
import sys

arguments_strModel = "CDFA"
arguments_strModelStateDict = './model/result_gcc_qnrf.pth'
arguments_strImg = './image/36.jpg'
arguments_strOut = './out/1_out.png'

arguments_intDevice = 0

mean_std =  ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

for strOption, strArgument in \
getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--device' and strArgument != '': arguments_intDevice = int(strArgument)  # device number
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # model type
    if strOption == '--model_state' and strArgument != '': arguments_strModelStateDict = strArgument  # path to the model state
    if strOption == '--img_path' and strArgument != '': arguments_strImg = strArgument  # path to the image
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored

torch.cuda.set_device(arguments_intDevice)

def evaluate(img_path, save_path):
    if arguments_strModel == "CDFA":
        net = CDFA(arguments_strModel)
        net.load_state_dict(torch.load(arguments_strModelStateDict, map_location=lambda storage, loc: storage), strict=False)
        net.cuda()
        net.eval()
    else:
        raise ValueError('Network cannot be recognized. Please define your own Network here.')
    
    img = Image.open(img_path)

    if img.mode == "L":
        img = img.convert('RGB')

    img = img_transform(img)
    with torch.no_grad():
        img = Variable(img[None, :, :, :]).cuda()
        pred_map = net.test_forward(img)

    pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
    pred = np.sum(pred_map)/100.0
    pred_map = pred_map/np.max(pred_map+1e-20)

    print("count result is {}".format(pred))

    den_frame = plt.gca()
    plt.imshow(pred_map, 'jet')
    plt.colorbar()
    den_frame.axes.get_yaxis().set_visible(False)
    den_frame.axes.get_xaxis().set_visible(False)
    den_frame.spines['top'].set_visible(False) 
    den_frame.spines['bottom'].set_visible(False) 
    den_frame.spines['left'].set_visible(False) 
    den_frame.spines['right'].set_visible(False) 
    plt.savefig(save_path, bbox_inches='tight',pad_inches=0,dpi=150)
    plt.close()

    print("save pred density map in {} success".format(arguments_strOut))

    print("end")


if __name__ == '__main__':
    evaluate(arguments_strImg, arguments_strOut)
