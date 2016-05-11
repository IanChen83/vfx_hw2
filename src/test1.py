import imgtools
import match
from PIL import Image
a = Image.open('../DSC_0069.JPG')
b = Image.open('../DSC_0070.JPG')

img, mapping = imgtools.showcase_pairing(a, b)

mat69 = match.parse_mat('../data/69.mat')
mat70 = match.parse_mat('../data/70.mat')
m = match.match(mat69, mat70)

imgtools.paste(a, b, m)

for x in m.queue:
    img_tools.mark_pair_for_concat(img, x, mapping)

#img.show()
