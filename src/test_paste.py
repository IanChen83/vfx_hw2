from imgtools import *
from PIL import Image
from match import parse_mat, match
import numpy as np
import sys

focal = sys.argv[1]

merger = Merger()
a = Image.open("../DSC_0069.JPG")
b = Image.open("../DSC_0070.JPG")
mat69 = parse_mat("../data/69.mat")
mat70 = parse_mat("../data/70.mat")

b2 = cyl_mapping(b, int(focal), mat70)
a2 = cyl_mapping(a, int(focal), mat69)
mat, queue = match(mat69, mat70)
for m in queue:
    mark_point(a2, m.fp1, 20)
    mark_point(b2, m.fp2, 20)

merger.add_image(a2, "a")
merger.add_image(b2, "b")

merger.add_matrix(mat, "a", "b")

merger.merge("a").save("{0}.jpg".format(focal))
