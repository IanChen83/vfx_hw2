from match import *
from LimitedPriorityQueue import LimitedPriorityQueue
from itertools import product, starmap, groupby
from imgtools import *
from PIL import Image
import numpy as np
import random
import threading
import sys
import os
focal = int(sys.argv[1])

merger = Merger()

matrices = dict()

name = "Gym"
end = 7
start = 0


class myThreadAddImage (threading.Thread):
    def __init__(self, start, end):
        threading.Thread.__init__(self)
        self.a = start
        self.b = end

    def run(self):
        add_image(self.a, self.b)


class myThreadAddMatrix (threading.Thread):
    def __init__(self, start, end):
        threading.Thread.__init__(self)
        self.a = start
        self.b = end

    def run(self):
        add_matrix(self.a, self.b)


def add_image(start, end):
    for i in range(start, end):
        merger.add_image(cyl_mapping(Image.open("../{name}/{num}.jpg".format(name=name, num=i)).convert('RGBA'), focal, matrices[i]), str(i))


def add_matrix(start, end):
    for i in range(start, end):
        mat, q = match(matrices[i - 1], matrices[i])
        merger.add_matrix(mat, str(i - 1), str(i))
        #showcase_pairing(merger.images[str(i-1)], merger.images[str(i)], q).save(name + "_show.jpg")

for i in range(start, end):
    matrices[i] = parse_mat("../{name}/{num}.mat".format(name=name, num=i))

# ========= ADD IMAGE =========
t=[]
tnum = 5
for i in range(tnum):
    t.append(myThreadAddImage(int(start + i * (end - start) / tnum), int(start + (i + 1) * (end - start) / tnum)))

for tx in t:
    tx.start()

for tx in t:
    tx.join()

# ========= ADD MATRIX =========
t2=[]
for i in range(tnum):
    t2.append(myThreadAddMatrix(int(start + i * (end - start - 1) / tnum + 1), int(start + (i + 1) * (end - start - 1) / tnum + 1)))

for tx in t2:
    tx.start()

for tx in t2:
    tx.join()

merger.merge(str(start)).save("{name}{f}.jpg".format(name=name, f=focal))
os.system("eog {name}{f}.jpg".format(name=name, f=focal))
