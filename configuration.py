from PIL import Image

# Image preprocessing
binary_threshold = 10

# Connected components
connection_radius = 2

# Latice and region creation
lattice_radius = 25
lattice_stride = 25
min_component_size = 4

# Simulation
num_free_cells = 50
neighborhood = 4
max_iterations = 50
num_simulations = 100

# Meta
verbose = False

# image = Image.open("images/binary.jpg")
# image = Image.open("images/Cap-3-20x-vessels-blackandwhite.tif")
image = Image.open("images/Art-8-vessels-blackandwhite 20x.tif")
# image = Image.open("images/Cap-2-vessels-blackandwhite-40X.tif")
