from PIL import Image
from imgtools import cyl_mapping

a = Image.open("../DSC_0070.JPG")
b = cyl_mapping(a, 3000)
b.save("cyl.jpg")
