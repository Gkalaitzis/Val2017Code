#Code "repurposed" from the Fordham Computational Neuroscience Lab and MIT CSAIL
#This code will store the response from the top layer of resnet18 trained on P365 in a csv

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import pandas as pd
import numpy

centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


#You can change the model, just make sure to use the proper .pth weight file as well.

arch = 'xxx'

#http://places2.csail.mit.edu/models_places365/{MODEL_NAME}_places365.pth.tar for checkpoint, put your model name in the space: resnet18, alexnet, resnet50 etc

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load("xxx", map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

maxVec=[]
responseMat=np.zeros((5000,365))

for i in range(1):
    
    #replace imPath and the path in files with your own chosen path
    
    imPath="xxx"
    files=os.listdir(imPath)
    k=0
    fileList=[]
    for file in files:
        img=Image.open(imPath+'/'+file)
        if numpy.ndim(img) < 3:
            continue
        input_img = V(centre_crop(img).unsqueeze(0))
        out=model.forward(input_img)
        outNP=out.detach().numpy()
        maxVec.append(out.argmax())
        responseMat[k,:]=outNP
        fileList.append(file)
        k=k+1

responseMat=responseMat[:(k),:]
df=pd.DataFrame(responseMat)
df.index=fileList
df.to_csv('netResponses.csv')
