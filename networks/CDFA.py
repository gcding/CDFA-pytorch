from torch.nn.modules import loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.autograd import Variable
from torchvision import models

class CDFA(nn.Module):
	def __init__(self, model_name):
		super(CDFA, self).__init__()
		self.model_name = model_name
		if model_name == "CDFA":
			from .vgg_sfcn_all import VGG_SFCN as net
			from .vgg_sfcn_all import Density_gen as density_gen
		else:
			raise ValueError('Network cannot be recognized. Please define your own Network here.')

		self.CCN = net()
		self.CCN_gen = density_gen()

		self.CCN = self.CCN.cuda()
		self.CCN_gen = self.CCN_gen.cuda()

		print("Model {} init success".format(model_name))

	def test_forward(self, timgv):
		if self.model_name == "CDFA":
			tfeature, _ = self.CCN(timgv, "True")
			tpred_map, _ = self.CCN_gen(tfeature)
			return tpred_map
		else:
			raise ValueError('Network cannot be recognized. Please define your own Network here.')

	def forward(self, simgv, sgtv, timgv, iter):
		pass

	def build_loss(self):
		pass

	@property
	def loss(self):
		pass