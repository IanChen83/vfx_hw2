from math import sqrt
from PIL import Image
import numpy as np
import numpy.linalg as nl


class fppair:
    def __init__(self, fps):
        self.fp1, self.fp2 = fps
        self.norm = nl.norm(fps[0].desc - fps[1].desc)

    def __eq__(self, other):
        if not isinstance(other, fppair):
            return NotImplemented
        return (self.fp1 == other.fp1 and self.fp2 == other.fp2) or (self.fp1 == other.fp2 and self.fp2 == other.fp1)

    def __ne__(self, other):
        ret = self.__eq__(other)
        return not ret if ret is not NotImplemented else NotImplemented

    def __lt__(self, other):
        return self.norm < other.norm

    def __gt__(self, other):
        return self.norm > other.norm

    def __le__(self, other):
        return self.norm <= other.norm

    def __ge__(self, other):
        return self.norm >= other.norm

    def __str__(self):
        return "({0}), ({1}), norm={2}".format(self.fp1, self.fp2, self.norm)


class vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def array(self):
        return np.array([self.x, self.y])

    def tuple(self):
        return self.x, self.y

    def length(self):
        return sqrt(self.x * self.x + self.y * self.y)

    def __str__(self):
        return "({0}, {1})".format(self.x, self.y)

    def __add__(self, other):
        if type(other) == int or type(other) == float:
            return vector(self.x + other, self.y + other)
        return vector(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        if type(other) == int or type(other) == float:
            return vector(self.x + other, self.y + other)
        return NotImplemented

    def __sub__(self, other):
        if type(other) == int or type(other) == float:
            return vector(self.x - other, self.y - other)
        return vector(self.x - other.x, self.y - other.y)

    def __rsub__(self, other):
        if type(other) == int or type(other) == float:
            return vector(self.x - other, self.y - other)
        return NotImplemented

    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            return vector(self.x * other, self.y * other)
        return self.x * other.x + self.y * other.y

    def __rmul__(self, other):
        if type(other) == int or type(other) == float:
            return vector(self.x * other, self.y * other)
        return NotImplemented

    def __truediv__(self, other):
        if type(other) == int or type(other) == float:
            return vector(self.x / other, self.y/other)
        return NotImplemented

    def __rtruediv__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        if type(other) == int or type(other) == float:
            self.x += other
            self.y += other
            return self
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):
        if type(other) == int or type(other) == float:
            self.x -= other
            self.y -= other
            return self
        self.x -= other.x
        self.y -= other.y
        return self

    def __imul__(self, other):
        if type(other) == int or type(other) == float:
            self.x *= other
            self.y *= other
            return self
        return NotImplemented

    def __itruediv__(self, other):
        if type(other) == int or type(other) == float:
            self.x /= other
            self.y /= other
            return self
        return NotImplemented

    def __neg__(self):
        return vector(-self.x, -self.y)

    def __abs__(self):
        return self.length()


class fpoint:
    def __init__(self, x, y, desc=[[0]], normalize=0):
        self.x = x
        self.y = y
        self.desc = desc

        if normalize != 0:
            self.desc = self.desc / nl.norm(self.desc) * normalize

    def norm(self):
        return nl.norm(self.desc)

    def xy(self):
        return self.x, self.y

    def array(self):
        return [self.x, self.y]

    def array3d(self):
        return [self.x, self.y, 1]

    def __str__(self):
        return "({0}, {1})".format(self.x, self.y)

    def __radd__(self, other):
        return NotImplemented

    def __add__(self, other):
        return vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return vector(int(self.x - other.x), int(self.y - other.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def transform(self, mat):
        ret = np.dot(mat, [[self.x], [self.y], [1]])
        return ret[0][0], ret[1][0]

    @staticmethod
    def near(a, b, dis):
        return abs(a.x - b.x) <= dis and abs(a.y - b.y) <= dis


class MatrixWrapper:
    def __init__(self, merger, mat, name1, name2):
        self.mat = mat
        self.img1 = merger.images.get(name1)
        self.img2 = merger.images.get(name2)

    def __str__(self):
        return "Matrix from {0} to {1}".format(self.img1.name, self.img2.name)


class ImageWrapper:
    def __init__(self, image, name):
        self.visited = False
        self.image = image
        self.name = name
        self.bond = None
        self.real_bond = None
        self.total_mat = None
        self.fanin = dict()
        self.fanout = dict()
        self.mask = None

    def __str__(self):
        return "Image('{0}')".format(self.name)
