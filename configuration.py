from PIL import Image

# Image preprocessing
binary_threshold = 10

# Connected components
connection_radius = 2

# Latice and region creation
lattice_radius = 50
lattice_stride = 50
min_component_size = 2

# Simulation
num_free_cells = 50
neighborhood = 1
max_iterations = 10
num_simulations = 100

# Meta
verbose = True
showplots = True

# image = Image.open("images/binary.jpg")
# image = Image.open("images/Cap-3-20x-vessels-blackandwhite.tif")
image = Image.open("images/Cap-2-vessels-blackandwhite-40X.tif")
