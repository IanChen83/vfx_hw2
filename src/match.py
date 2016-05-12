import scipy.io as si
import random
from ImageClass import *
from itertools import groupby, product, permutations, combinations
from LimitedPriorityQueue import *
from imgtools import *

VOTING_FILTER_ENABLE = False
RANSAC_SAMPLE_NUM = 3
RANSAC_THRESHOLD = 5

def fpoint_hash_func_norm(x, y, desc):
    return int(floor(nl.norm(desc)/100))


def parse_mat(path):
    temp = si.loadmat(path)
    mat = temp["descriptor"].transpose(2, 0, 1)
    points = temp["coor"].astype(np.int32)

    ret = []

    for i in range(len(points)):
        ret.append(fpoint(int(points[i][0]), int(points[i][1]), mat[i], 1000))
    print("Parse {0} points from {1}".format(len(ret), path))
    return ret


def match(target, ref):
    ########### Method 1 #############
    #q3 = []
    #for x in target:
    #    ret = LimitedPriorityQueue(1)
    #    for y in ref:
    #        ret.push(fppair((x,y)))
    #    q3 = q3 + ret.queue

    #return ransac(q3, 100), q3

    ########### Method 2 #############
    ret = LimitedPriorityQueue(30)
    for x in product(target, ref):
        ret.push(fppair(x))
    # return ret.queue
    q3 = voting_filter(ret.queue)
    if len(q3) < RANSAC_SAMPLE_NUM:
        print("Voting filter failed")
        return ransac(ret.queue, RANSAC_THRESHOLD), q3
    return ransac(ret.queue, RANSAC_THRESHOLD), ret.queue

    # Method 1: Average #
    #mat = np.identity(3)

    # Translation
    #sx, sy = 0, 0
    #for pp in q3:
    #    sx += (pp.fp1.x - pp.fp2.x)
    #    sy += (pp.fp1.y - pp.fp2.y)

    #sx /= len(q3)
    #sy /= len(q3)

    #mat[0, 2] = sx
    #mat[1, 2] = sy

    # Rotation (not working :(( )
    # r = 0
    # count = 0
    # for v in combinations(q3, 2):
    #     v1 = v[0].fp1 - v[1].fp1
    #     v2 = v[0].fp2 - v[1].fp2
    #     _cos = (v1.x * v2.x + v1.y * v2.y) / (v1.length() * v2.length())
    #     r += acos(_cos)
    #     count += 1
    # c = cos(r/count)
    # s = sin(r/count)
    #
    #
    # mat[0,0] = c
    # mat[0,1] = s
    # mat[1,0] = -s
    # mat[1,1] = c

    #print("Matching with inlier cost = {0}".format(avg_distance(q3, mat)))

    #return mat, q3


def distance_cost(fpp, mat):
    fp1 = fpp.fp1
    m = fpp.fp2.transform(mat)
    # return sqrt((m[0] - fp2.x) * (m[0] - fp2.x) + (m[1] - fp2.y) * (m[1] - fp2.y))
    return abs(m[0] - fp1.x) + abs(m[1] - fp1.y)


def avg_distance(queue, mat, thr=0):
    count = 0
    cost = 0
    q = []
    if thr == 0:
        for pp in queue:
            count += 1
            cost += distance_cost(pp, mat)

        return 0, cost/count
    else:
        inl, outl = 0, 0
        for pp in queue:
            c = distance_cost(pp, mat)
            count += 1
            if c >= thr:
                outl += 1
            else:
                inl += 1
                cost += c
        if inl != 0:
            return inl, cost/inl
        else:
            return inl, 1000000


def ransac(fpps, thr, k=5000):
    gn = RANSAC_SAMPLE_NUM
    max_inl, min_distance, min_mat =0, 1000000, None
    for _ in range(k):
        g = random.sample(fpps, gn)
        v = sum([pp.fp1 - pp.fp2 for pp in g])/gn
        mat = np.identity(3)
        mat[0, 2] = v.x
        mat[1, 2] = v.y
        inl, avg = avg_distance(fpps, mat, thr)
        if inl > max_inl:
            max_inl = inl
            min_mat = mat
#        if avg < min_distance:
#            min_distance, min_mat = avg, mat
    print("max inl: {0}".format(max_inl))
#    print(min_distance)
    if min_mat is None:
        return ransac(fpps, thr + 150, k)
    return min_mat


def voting_filter(q):
    if VOTING_FILTER_ENABLE == False:
        return q
    key1x = lambda x: x.fp1.x
    key1y = lambda x: x.fp1.y
    key2x = lambda x: x.fp2.x
    key2y = lambda x: x.fp2.y

    q2 = []
    q3 = []
    for x, gx in groupby(sorted(q, key=key1x), key=key1x):
        for y, gy in groupby(sorted(gx, key=key1y), key=key1y):
            g = list(gy)
            if len(g) == 1:
                q2 = q2 + g
    for x, gx in groupby(sorted(q2, key=key2x), key=key2x):
        for y, gy in groupby(sorted(gx, key=key2y), key=key2y):
            g = list(gy)
            if len(g) == 1:
                q3 = q3 + g
    return q3
