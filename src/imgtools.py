from PIL import Image, ImageDraw
from math import floor, sqrt, atan
import numpy as np
import itertools as it
from ImageClass import MatrixWrapper, ImageWrapper


def mark_point(img, xy, size=6):
    draw = ImageDraw.Draw(img)
    x = xy[0]
    y = xy[1]
    fill = (255, 0, 0)
    s = int(size/2)
    draw.rectangle([x-s, y-s, x+s, y+s], fill=fill, outline=fill)
    del draw
    return img


def showcase_mask(mask, h=300):
    Image.fromarray(np.uint8(np.repeat(mask, h).reshape((mask.shape[0], h)).transpose(1, 0)), mode='L').show()


def showcase_pairing(img1, img2, pp):
    size1, size2 = img1.size, img2.size
    ratio = 0.0
    w_over_h = ((size1[0] + size2[0])/(size1[1] + size2[1]) > (16/9))
    if w_over_h:
        ratio = 1920/(size1[0] + size2[0])
        ret = Image.new('RGB', (int(max(size1[0], size2[0]) * ratio), int(ratio * (size1[1] + size2[1]))))
    else:
        ratio = 1080/(size1[1] + size2[1])
        ret = Image.new('RGB', (int((size1[0] + size2[0]) * ratio), int(max(size1[1], size2[1]) * ratio)))

    size1 = (int(size1[0] * ratio), int(size1[1] * ratio))
    size2 = (int(size2[0] * ratio), int(size2[1] * ratio))
    shifter = (size1[0], 0)

    nim1 = img1.resize(size1)

    nim2 = img2.resize(size2)

    ret.paste(nim1, (0, 0))
    ret.paste(nim2, shifter)

    def mapping(p, is2):
        if is2:
            return int(p[0] * ratio) + size1[0], int(p[1] * ratio)
        else:
            return int(p[0] * ratio), int(p[1] * ratio)
    for pair in pp:
        draw = ImageDraw.Draw(ret, 'RGBA')
        x1, y1 = mapping(pair.fp1.xy(), False)
        x2, y2 = mapping(pair.fp2.xy(), True)
        fill = (255, 0, 0, 60)
        draw.line([(x1, y1), (x2, y2)], width=4, fill=fill)
    return ret


def cyl_function(focal):
    def _x(tx, ty):
        return int(focal * atan(tx / focal))

    def _y(tx, ty):
        return int(focal * ty / sqrt(focal * focal + tx * tx))
    return _x, _y


def transform(p, mat):
    ret = np.dot(mat, [[p[0]], [p[1]], [1]])
    return ret[0][0], ret[1][0]


def cyl_mapping(img, focal, ps=None):
    x, y = img.size
    x_start = -int(floor(x/2))
    y_start = -int(floor(y/2))
    pimg = img.load()

    _x, _y = cyl_function(focal)

    pret = np.array(Image.new("RGB", img.size)).transpose((1, 0, 2))

    for t in it.product(range(x_start, x + x_start), range(y_start, y + y_start)):
        pret[_x(*t) - x_start, _y(*t) - y_start] = pimg[t[0] - x_start, t[1] - y_start]

    if ps is None:
        return pret

    for fp in ps:
        fp.x, fp.y = _x(fp.x + x_start, fp.y + y_start) - x_start, _y(fp.x + x_start, fp.y + y_start) - y_start

    return pret


class Merger:
    def __init__(self):
        self.images = dict()
        self.output = None
        self.output_mask = None

    @staticmethod
    def get_overlap_x(bond1, bond2):
        print("bond1:{0}".format(bond1))
        print("bond2:{0}".format(bond2))
        # return overlap_y, if bond2 is in the right side of bond1

        def seg3(seg, p):
            if p < seg[0]:
                return 0
            elif p > seg[1]:
                return 2
            else:
                return 1

        a = seg3((bond1[0], bond1[2]), bond2[0])
        if a == 0:
            if bond2[2] > bond1[0]:
                return [bond1[0], bond2[2]], False
            return None, False
        elif a == 1:
            return [bond2[0], bond1[2]], True
        else:
            return None, True

    @staticmethod
    def within(bond, p):
        return p[0] > bond[0] and p[1] > bond[1] and p[0] < bond[2] and p[1] < bond[3]

    def add_matrix(self, mat, name1, name2):
        print("Add matrix ({0}, {1})".format(name1, name2))
        ret = MatrixWrapper(self, mat, name1, name2)

        if ret.img1 is None or ret.img2 is None:
            raise ValueError

        ret.img1.fanout[name2] = ret
        ret.img2.fanin[name1] = ret

    def add_image(self, img, name):
        print("Add image with name: " + name)
        if self.images.get(name) is None:
            self.images[name] = ImageWrapper(img, name)
        else:
            print("Warning: img with name {0} is replaced".format(name))
            im = self.images[name]
            im.image = img
            im.name = name

    @staticmethod
    def get_transform_boundary(p, mat):
        a, b, x, y = p
        oldbond = [[a, a, x, x], [b, y, b, y], [1, 1, 1, 1]]
        newbond = np.dot(mat, oldbond)
        ret = min(newbond[0]), min(newbond[1]), max(newbond[0]), max(newbond[1])
        return ret

    @staticmethod
    def dfs_default_func(link):
        return

    def dfs(self, root, visit_func, visited_func):
        # First node
        rnode = self.images[root]
        rnode.visited = True
        stack = list(rnode.fanout.values())
        cur = stack.pop()
        # Start DFS
        while cur is not None:
            if cur.img2.visited is True:
                if visited_func is not None:
                    visited_func(cur)
            else:
                if visit_func is not None:
                    visit_func(cur)
                cur.img2.visited = True
                stack = stack + list(cur.img2.fanout.values())
            if len(stack) == 0:
                cur = None
            else:
                cur = stack.pop()
        self.clear_visited()

    def clear_visited(self):
        for im in self.images.values():
            im.visited = False

    @staticmethod
    def draw_value(mat, x, v):
        mat[x] = min(v, mat[x])

    @staticmethod
    def draw_gradient(mask, x1, x2, v1, v2):
        s = (v2 - v1) / (x2 - x1)
        v = v1
        for y in range(x1, x2):
            Merger.draw_value(mask, y, v)
            v += s

    @staticmethod
    def blend_mask(img1, img2):
        seg, bottom = Merger.get_overlap_x(img1.bond, img2.bond)
        assert bottom is True
        if seg is None:
            return

        shift1, shift2 = img1.bond[0], img2.bond[0]

        print(seg)
        # For img1
        for x in range(int(seg[0] - shift1)):
            Merger.draw_value(img1.mask, x, 255)
        Merger.draw_gradient(img1.mask, int(seg[0] - shift1), int(seg[1] - shift1), 255, 1)
        for x in range(int(seg[1] - shift1), img1.image.shape[0]):
            Merger.draw_value(img1.mask, x, 1)
        # For img2
        for x in range(int(seg[0] - shift2)):
                Merger.draw_value(img2.mask, x, 1)
        Merger.draw_gradient(img2.mask, int(seg[0] - shift2), int(seg[1] - shift2), 1, 255)
        for x in range(int(seg[1] - shift2), img2.image.shape[0]):
            Merger.draw_value(img2.mask, x, 255)

    def merge(self, root):
        def prepare_merge(link):
            a = link.img1
            b = link.img2

            print("?")
            b.total_mat = np.dot(link.mat, a.total_mat)
            b.bond = Merger.get_transform_boundary((0, 0, b.image.shape[0], b.image.shape[1]), b.total_mat)
            b.mask = np.ones(b.image.shape[0]) * 255

        # DFS to find boundary and total matrix
        # root node
        iimg = self.images[root]
        iimg.total_mat = np.identity(3)
        iimg.bond = 0, 0, iimg.image.shape[0], iimg.image.shape[1]
        iimg.mask = np.ones(iimg.image.shape[0]) * 255

        self.dfs(root, prepare_merge, None)

        top, bottom, left, right = 0, 0, 0, 0
        for im in self.images.values():
            top = min(int(im.bond[1]), top)
            bottom = max(int(im.bond[3]), bottom)
            left = min(int(im.bond[0]), left)
            right = max(int(im.bond[2]), right)

        shiftx = -left
        shifty = -top
        self.output = np.zeros((int(right - left), int(bottom - top), 3))
        self.output_mask = np.ones(int(right - left))

        merge_list = list(sorted(self.images.values(), key=lambda x: x.bond[0]))
        for i in range(len(merge_list)):
            im = merge_list[i]
            print("===== {0} =====".format(im.name))
            print("Boundary: {0}".format(im.bond))

            if i != len(merge_list) - 1:
                Merger.blend_mask(merge_list[i], merge_list[i+1])

            self.paste(im.image, im.total_mat, (shiftx, shifty), im.mask)

        print(self.output_mask)

        self.output /= self.output_mask[:, np.newaxis, np.newaxis]

        return Image.fromarray(np.uint8(self.output.transpose(1, 0, 2)))

    def paste(self, b, mat, shift=(0, 0), mask=None):
        if mask is not None:
            assert b.shape[0] == mask.shape[0]
            shift = int(mat[0, 2] + shift[0]), int(mat[1, 2] + shift[1])
            self.output_mask[shift[0]: shift[0] + mask.shape[0]] += mask
            Merger.paste_matrix(self.output, b * mask[:, np.newaxis, np.newaxis], shift)
        else:
            shift = int(mat[0, 2] + shift[0]), int(mat[1, 2] + shift[1])
            Merger.paste_matrix(self.output, b, shift)

    @staticmethod
    def paste_matrix(mat1, mat2, coor):
        size_x, size_y, layer = np.shape(mat2)
        coor_x, coor_y = coor
        end_x, end_y = (coor_x + size_x), (coor_y + size_y)
        mat1[coor_x:end_x, coor_y:end_y] = mat1[coor_x:end_x, coor_y:end_y] + mat2
        return mat1
