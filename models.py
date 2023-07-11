import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
from GradRev import RevGrad
from torch.distributions.uniform import Uniform
import functools

class IEncoder(nn.Module):
    
  def __init__(self):
    super(IEncoder, self).__init__()   
    self.input_channel=1
    self.inter_channel=64
    self.conv1=nn.Sequential(nn.Conv2d(self.input_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))
    self.layer1=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))
    self.pool1=nn.MaxPool2d(kernel_size=(2, 2))
    self.layer2=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))
    self.pool2=nn.MaxPool2d(kernel_size=(2, 2))
    self.layer3=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
   			                 nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))           
                            

  def forward(self,x):
      x=self.conv1(x)
      x1=self.layer1(x)
      x=self.pool1(x1)
      x2=self.layer2(x)
      x=self.pool2(x2)
      x=self.layer3(x)
      return x, x1, x2
  
    
class IDecoder(nn.Module):
    
  def __init__(self, image_size):
    super(IDecoder, self).__init__()   
    self.input_channel=1
    self.inter_channel=64               
    self.image_size=image_size                       
    self.layer1=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True), 
                            
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True), 
                            
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))
    
    self.pool1=nn.Upsample(scale_factor=2, mode='nearest')

    self.layer2=nn.Sequential(nn.Conv2d(2*self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))

    self.pool2=nn.Upsample(scale_factor=2, mode='nearest')    
    
    self.layer3=nn.Sequential(nn.Conv2d(2*self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                             nn.ReLU(inplace=True))
    self.conv=nn.Conv2d(self.inter_channel,self.input_channel,3,padding=1)

  def forward(self,x,x1,x2):
      x=self.layer1(x)
      x=self.pool1(x)
      xd1=self.layer2(torch.cat((x2 , x),1))
      x=self.pool2(xd1)
      xd2=self.layer3(torch.cat((x1 , x),1))
      x=self.conv(xd2)
      return x

class DomainClassifier(nn.Module):
    def __init__(self, image_size, inter_channel):
        super(DomainClassifier, self).__init__()
        self.image_size=image_size
        self.inter_channel=inter_channel
        self.layer1=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                               nn.ReLU(inplace=True),
                                nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                                nn.ReLU(inplace=True)
                             )
        self.layer2=nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                                 nn.ReLU(inplace=True),
                                  nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                                  nn.ReLU(inplace=True),
                                   nn.Conv2d(self.inter_channel,self.inter_channel,3,padding=1),
                                   nn.ReLU(inplace=True)
                                 )
    
        self.classifier=nn.Sequential(nn.Linear(int(self.image_size*self.image_size*self.inter_channel), 512),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(512, 2))
    def forward(self,x):
        x= self.layer1(x)
        x= self.layer2(x)
        x=x.view(-1,self.image_size*self.image_size*self.inter_channel)
        out=self.classifier(x)
        return out
    
    

class AuxDenoiser(nn.Module):
    def __init__(self, image_size):
        super(AuxDenoiser, self).__init__()
        self.image_size=image_size
        self.inter_channel=64
        self.input_channel=1
        
        self.encoder=IEncoder()
        self.decoder=IDecoder(image_size)
        self.auxdecoder1=IDecoder(image_size)
        self.auxdecoder2=IDecoder(image_size)
        
        self.classifier=DomainClassifier(int(self.image_size/4), self.inter_channel)    
        self.gradrev1=RevGrad()
        self.pool=nn.MaxPool2d(2, stride=2)
        self.uni_dist1 = Uniform(-0.3, 0.3)

    def forward(self, x):
        x,x1,x2=self.encoder(x)
        out =self.decoder(x,x1,x2)
        
        noise_vector = self.uni_dist1.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        noise_vector = self.uni_dist1.sample(x1.shape[1:]).to(x1.device).unsqueeze(0)
        x1_noise = x1.mul(noise_vector) + x1
        noise_vector = self.uni_dist1.sample(x2.shape[1:]).to(x2.device).unsqueeze(0)
        x2_noise = x2.mul(noise_vector) + x2
        out_aux1= self.auxdecoder1(x_noise,x1_noise,x2_noise)
        
        noise_vector = self.uni_dist1.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        noise_vector = self.uni_dist1.sample(x1.shape[1:]).to(x1.device).unsqueeze(0)
        x1_noise = x1.mul(noise_vector) + x1
        noise_vector = self.uni_dist1.sample(x2.shape[1:]).to(x2.device).unsqueeze(0)
        x2_noise = x2.mul(noise_vector) + x2
        out_aux2= self.auxdecoder2(x_noise,x1_noise,x2_noise)
                     
        x_c=self.gradrev1(x)
        logits=self.classifier(x_c)     
        return out, logits, out_aux1, out_aux2


class AuxDenoiserTest(nn.Module):
    def __init__(self, image_size):
        super(AuxDenoiserTest, self).__init__()
        self.image_size=image_size
        self.inter_channel=64
        self.input_channel=1
        
        self.encoder=IEncoder()
        self.decoder=IDecoder(image_size)

    def forward(self, x):
        x,x1,x2=self.encoder(x)
        out =self.decoder(x,x1,x2)
        return out