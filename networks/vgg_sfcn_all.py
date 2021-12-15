import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class convDU(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(9,1)
        ):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).resize(n,c,1,w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)
            # pdb.set_trace()
            # fea[:,i,:,:] = self.conv(fea[:,i-1,:,:].expand(n,1,h,w))+fea[:,i,:,:].expand(n,1,h,w)


        for i in range(h):
            pos = h-i-1
            if pos == h-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]
        # pdb.set_trace()
        fea = torch.cat(fea_stack, 2)
        return fea

class convLR(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(1,9)
        ):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).resize(n,c,h,1)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)

        for i in range(w):
            pos = w-i-1
            if pos == w-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]


        fea = torch.cat(fea_stack, 3)
        return fea

class VGG_SFCN(nn.Module):
	def __init__(self, load_weights=False):
		super(VGG_SFCN, self).__init__()
		self.seen = 0
		self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
		self.backend_feat  = [512, 512, 512,256,128,64]
		self.frontend = make_layers(self.frontend_feat)
		self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
		
		self.convDU = convDU(in_out_channels=64,kernel_size=(1,9))
		self.convLR = convLR(in_out_channels=64,kernel_size=(9,1))

		self.avgpool = nn.AdaptiveAvgPool2d((3,3))

	def forward(self,x, mmd=False):
		x = self.frontend(x)
		x = self.backend(x)
		
		x = self.convDU(x)
		x = self.convLR(x)

		if mmd:
			mmd = self.avgpool(x)
			mmd = torch.flatten(mmd, 1)
			return x, mmd
		return x


class Density_gen(nn.Module):
	def __init__(self):
		super(Density_gen, self).__init__()
		self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
		self.bri = nn.Conv2d(64,1, kernel_size=1)
		self.acti = nn.ReLU()


	def forward(self, input):
		out = self.output_layer(input)
		bri = self.bri(input)
		out = out - bri
		out = self.acti(out)
		out = F.upsample(out,scale_factor=8, mode='bicubic')

		return out, bri


class Global_domain(nn.Module):
	def __init__(self):
		super(Global_domain, self).__init__()
		# coming soon
		pass
	def forward(self, global_cri, global_iter_num):
		# coming soon
		pass


class Local_domain(nn.Module):
	def __init__(self):
		super(Local_domain, self).__init__()
		# coming soon
		pass

	def forward(self, csr_feature, local_iter_num):
		# coming soon
		pass
				
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
	if dilation:
		d_rate = 2
	else:
		d_rate = 1
	layers = []
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)                


