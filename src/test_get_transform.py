from imgtools import get_transform, fppair, fpoint
from math import sqrt
def case1():
    fpp1 = fppair((
        fpoint(0,0,0),
        fpoint(0,0,0)
            ))
    fpp2 = fppair((
        fpoint(0,2,0),
        fpoint(sqrt(2),sqrt(2),0)
            ))
    fpp3 = fppair((
        fpoint(2,0,0),
        fpoint(sqrt(2),-sqrt(2),0)
            ))

    m = get_transform(fpp1, fpp2, fpp3)

    print("Transformation Matrix:\n{0}".format(m))

    # Test case
    print("Transform {0} to {1}".format(fpp1.fp1, fpp1.fp1.transform(m)))
    print("Transform {0} to {1}".format(fpp2.fp1, fpp2.fp1.transform(m)))
    print("Transform {0} to {1}".format(fpp3.fp1, fpp3.fp1.transform(m)))

    fp4 = fpoint(4*sqrt(2), 0, [[0]])
    print("Transform {0} to {1}".format(fp4, fp4.transform(m)))

    return True

def case2():
    fpp1 = fppair((
        fpoint(0,0,0),
        fpoint(1,2,0)
            ))
    fpp2 = fppair((
        fpoint(0,2,0),
        fpoint(1 + sqrt(2),2 + sqrt(2),0)
            ))
    fpp3 = fppair((
        fpoint(2,0,0),
        fpoint(1 + sqrt(2),2 - sqrt(2),0)
            ))

    m = get_transform(fpp1, fpp2, fpp3)

    print("Transformation Matrix:\n{0}".format(m))

    # Test case
    print("Transform {0} to {1}".format(fpp1.fp1, fpp1.fp1.transform(m)))
    print("Transform {0} to {1}".format(fpp2.fp1, fpp2.fp1.transform(m)))
    print("Transform {0} to {1}".format(fpp3.fp1, fpp3.fp1.transform(m)))

    fp4 = fpoint(4*sqrt(2), 0, [[0]])
    print("Transform {0} to {1}".format(fp4, fp4.transform(m)))

    return True

if __name__ == "__main__":
    cases = [case1, case2]
    for c in cases:
        name = c.__name__
        print("========== Case '{0}' Begin ==========".format(name))
        if c() is True:
            print("Case '{0}' succeeds".format(name))
        else:
            print("Case '{0}' fails".format(name))
        print("========== Case '{0}' End ==========".format(name))
