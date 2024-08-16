import torch
import numpy
a = numpy.array([227., 490., 361., 28.])
a = torch.from_numpy(a)
b = torch.argmax(a)
pass