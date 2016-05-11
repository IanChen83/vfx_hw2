from imgtools import *
from PIL import Image
from match import parse_mat, match
import numpy as np
import sys

focal = 4000

merger = Merger()
a = Image.open("../../../Documents/VFX/hw2/data/DSC_0069.JPG")
b = Image.open("../../../Documents/VFX/hw2/data/DSC_0070.JPG")
mat69 = parse_mat("../../../Desktop/69.mat")
mat70 = parse_mat("../../../Desktop/70.mat")

for p in mat69:
    mark_point(a, p.xy())
for p in mat70:
    mark_point(b, p.xy())

a.save("69.jpg")
b.save("70.jpg")
