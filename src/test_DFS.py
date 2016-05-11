from imgtools import Merger
from PIL import Image
from match import parse_mat, match
merger = Merger()
a = Image.open("../DSC_0069.JPG")
mat69 = parse_mat("../data/69.mat")
mat70 = parse_mat("../data/70.mat")

mat = match(mat69, mat70)

merger.add_image(a, "1")
merger.add_image(a, "2")
merger.add_image(a, "3")
merger.add_image(a, "4")

merger.add_matrix(mat, "1", "2")
merger.add_matrix(mat, "2", "3")
merger.add_matrix(mat, "3", "1")
merger.add_matrix(mat, "3", "4")
merger.add_matrix(mat, "1", "4")

merger.dfs("1", Merger.dfs_print_func, None)
