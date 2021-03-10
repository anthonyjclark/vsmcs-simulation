#!/usr/bin/env python3

# TODO: Add probability of binding proportaional
# to push/pull and some global scale.

from collections import deque
from datetime import datetime
from math import inf
from os import getcwd, makedirs, path
from random import shuffle, randrange
from shutil import copyfile
from time import time
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation

import numpy as np
from numpy import logical_and, sqrt, uint, int64

from PIL import Image

# An adjacency list graph
Vertex = Tuple[int, int]
Graph = Dict[Vertex, List[Vertex]]

MOVING, STATIONARY, EMPTY = True, False, False


# TODO: for the love of GOD do not hard code this
neighborhood_distances = np.array(
    [
        [0.236, 0.277, 0.316, 0.333, 0.316, 0.277, 0.236],
        [0.277, 0.354, 0.447, 0.5, 0.447, 0.354, 0.277],
        [0.316, 0.447, 0.707, 1.0, 0.707, 0.447, 0.316],
        [0.333, 0.5, 1.0, 0.0, 1.0, 0.5, 0.333],
        [0.316, 0.447, 0.707, 1.0, 0.707, 0.447, 0.316],
        [0.277, 0.354, 0.447, 0.5, 0.447, 0.354, 0.277],
        [0.236, 0.277, 0.316, 0.333, 0.316, 0.277, 0.236],
    ]
)


def binarize_and_threshold_image(image: Image, threshold: int) -> np.ndarray:
    """Take any image with values in [0 255] and convert all values to 0 or 1.

    args
        image : an image with any number of channels and any value range
        threshold : all value below or equal to the threshold are set to 0

    return the binarized image as a numpy matrix
    """

    binary_image = image.convert("L")  # type: ignore

    binary_image_array = np.array(binary_image)

    binary_image_array[binary_image_array <= threshold] = 0
    binary_image_array[binary_image_array > threshold] = 1

    return binary_image_array


def create_matrix_from_adjacency_list(
    matrix: np.ndarray, radius: float, epsilon: float = 0.00001
) -> Graph:
    """Create an adjacency list graph from an image matrix.

    args
        matrix : an m-by-n matrix
        radius : minimum cell distance to consider between connected vertices
        epsilon : floating-point comparison threshold

    return adjacency list graph
    """

    adj_list = {}

    width, height = matrix.shape

    num_digits = len(str(height))

    # Consider each cell row
    for row in range(height):

        # Use radius to get the top (rt) and bottom (rb) bounds on the
        # neighboring rows
        rt = max(0, row - radius)
        rb = min(height, row + radius + 1)

        # All row index values in the range
        y = np.arange(rt, rb)

        # Consider each cell column
        for col in range(width):

            # Use radius to get the left (cl) and right (cr) bounds on the
            # neighboring columns
            cl = max(0, col - radius)
            cr = min(width, col + radius + 1)

            # All column index values in the range
            x = np.arange(cl, cr)

            # Color of the "current" cell
            cell_color = matrix[row, col]

            # All cell indices in the range
            xg, yg = np.meshgrid(x, y)

            # All cells within the threshold distance
            dists = sqrt((xg - col) ** 2 + (yg - row) ** 2)
            within_radius = dists <= (radius + epsilon)

            # All cells with the same color
            connect = logical_and(matrix[rt:rb, cl:cr] == cell_color, within_radius)

            # Update the adjcenecy matrix will all matching colors
            adj_list[(row, col)] = {(x + rt, y + cl) for (x, y) in np.argwhere(connect)}

        if VERBOSE and row % 100 == 0:
            print(f"Completed processing row {row:>{num_digits}} of {height-1}")

    return adj_list


def bfs(adj_list: Graph, start_vertex: Vertex) -> Set[Vertex]:
    """Breadth-first search for adjacency list.

    args
        adj_list : graph in adjacency list form
        start_vertex : beginning vertex for bfs

    return all connected vertices
    """

    found = {start_vertex}

    visit_queue = deque([start_vertex])

    while len(visit_queue) != 0:
        v_found = visit_queue.popleft()
        for v_other in adj_list[v_found]:
            if v_other not in found:
                found.add(v_other)
                visit_queue.append(v_other)

    return found


def find_connected_compnents(adj_list: Graph) -> List[Set[Vertex]]:
    """Compute all connected components of adjacency list.

    args
        adj_list : graph in adjacency list form

    return all components
    """

    components = []

    all_found = set()

    for vertex in adj_list:
        if vertex not in all_found:
            component = bfs(adj_list, vertex)
            components.append(component)
            all_found |= component

    return components


def round_to_stride(value: int, stride: int) -> int:
    return int(stride * round(value / stride))


def create_lattice_of_component_counts(
    input_shape: Tuple[int, int],
    radius: float,
    stride: int,
    components,
    min_cardinality: int,
) -> np.ndarray:
    """Create lattice where each cell indicates the number of components within radius.

    args
        input_shape : shape of the lattice
        radius : distance away from lattice cell to search for components
        stride : distance between lattice cells
        components : all components of the original graph
        min_cardinality : ignore all components with fewer than min_cardinality vertices

    return the lattice
    """

    max_row, max_col = input_shape

    # Size the lattice based on the input shape and the stride
    # (the plus one is to account for starting the stride at 0 and potentially
    # ending on the last column or row)
    lattice_height = max_row // stride
    lattice_width = max_col // stride
    lattice = np.zeros((lattice_height, lattice_width), dtype=uint)

    num_digits = len(str(len(components)))

    for (i, component) in enumerate(components):

        indices_to_skip = set()

        if len(component) >= min_cardinality:

            for (vy, vx) in component:

                # Radius around current component vertex taking
                # into account the stride
                rt = round_to_stride(max(0, vy - radius), stride)
                rb = min(max_row, vy + radius + 1)

                cl = round_to_stride(max(0, vx - radius), stride)
                cr = min(max_col, vx + radius + 1)

                for y in range(rt, rb, stride):
                    for x in range(cl, cr, stride):
                        if (y, x) not in indices_to_skip:
                            dist = sqrt((vx - x) ** 2 + (vy - y) ** 2)
                            if dist < radius:

                                # Convert the cell indices to lattice indices by
                                # considering the stride
                                try:
                                    lattice[y // stride, x // stride] += 1
                                except:
                                    # Ignore out of bounds as it is an artifact of starting
                                    # the stride at 0
                                    pass
                                indices_to_skip.add((y, x))

        if VERBOSE and i % 10 == 0:
            print(
                f"Completed processing component {i:>{num_digits}} of {len(components)-1}"
            )

    return lattice


def get_brightnesses_from_image(image: Image, stride: int) -> np.ndarray:
    """Compute scaled brightness for image.

    args
        image : image input
        stride : stride used for lattice (different function)

    return values caled in [-1 1] with the same shape as the lattice
    """

    mat = np.array(image)

    # Full height and width rounded down to max stride index
    fh, fw, _ = mat.shape
    fh = fh // stride * stride
    fw = fw // stride * stride

    # Scaled height and width
    sh, sw = fh // stride, fw // stride

    # Lop of edges of image if necessary and average over RGB channels
    mat = mat[:fh, :fw].mean(2)

    # Average each stride-by-stride box
    mat = mat.reshape((sh, stride, sw, stride)).mean(3).mean(1)

    # Scale to [-1 1]
    max_val = np.amax(mat)
    return mat / max_val * 2 - 1


def compute_region_scores(lattice: np.ndarray, brightness: np.ndarray) -> np.ndarray:
    """Compute region scores based on graph components and brightness.

    args
        lattice : lattice computed from graph components
        brightness : scaled values from the original image

    return
    """

    max_components = np.amax(lattice)

    regions = lattice - max_components / 2
    regions *= brightness
    return regions / np.abs(regions).max()


def travel(
    ycell: int,
    xcell: int,
    regions: np.ndarray,
    neighborhood: int,
    cell_grid: np.ndarray,
) -> Tuple[int, int]:
    """Move a VSMCS from its current location based on its neighborhood.

    args
        ycell : cell's current y index
        xcell : cell's current x index
        regions : grid of attraction/repulsion
        neighborhood : area in which to search for a new position
        cell_grid : current status of all cells

    return a new location for the cell
    """

    # # Score of the current cell
    # score = regions[ycell, xcell] if cell_grid[ycell, xcell] == EMPTY else inf

    ynew, xnew = ycell, xcell

    height, width = cell_grid.shape

    # Get neighborhood around this cell (clamped by borders of image)
    # (+ 1 for exclusive upper bound)
    ymin = max(0, ycell - neighborhood)
    ymax = min(height, ycell + neighborhood + 1)

    xmin = max(0, xcell - neighborhood)
    xmax = min(width, xcell + neighborhood + 1)

    # TODO: this is the only method wherein we moved to the local optimum
    # for y in range(ymin, ymax):
    #     for x in range(xmin, xmax):

    #         neigh_score = regions[y, x]

    #         # TODO: Change score based on distance from current cell
    #         # TODO: Compute a direction based on all neighbor values
    #         # (not just peak value)

    #         if neigh_score < score and cell_grid[y, x] == EMPTY:
    #             score = neigh_score
    #             ynew, xnew = y, x

    # # Get offset indices
    # xoffmin = -min(xcell, neighborhood)
    # xoffmax = min(width - 1 - xcell, neighborhood)

    # yoffmin = -min(ycell, neighborhood)
    # yoffmax = min(height - 1 - ycell, neighborhood)

    # xoffsets = range(xoffmin, xoffmax + 1)
    # yoffsets = range(yoffmin, yoffmax + 1)

    numleft = min(neighborhood, xcell)
    numright = min(neighborhood, width - (xcell + 1))

    numabove = min(neighborhood, ycell)
    numbelow = min(neighborhood, height - (ycell + 1))

    xdirfactor = [-1] * numleft + [0] + [1] * numright
    ydirfactor = [-1] * numabove + [0] + [1] * numbelow

    xs, ys = np.meshgrid(xdirfactor, ydirfactor)

    # Negate scores so that the direction is toward negative values
    neigh_scores = -regions[ymin:ymax, xmin:xmax] / 4

    # Truncate neighborhood_distances
    if xcell - neighborhood < 0:
        sx = abs(xcell - neighborhood)
    else:
        sx = 0

    if (xcell + neighborhood) > (width - 1):
        ex = (neighborhood * 2 + 1) - abs(xcell + neighborhood - width + 1)
    else:
        ex = neighborhood * 2 + 1

    if ycell - neighborhood < 0:
        sy = abs(ycell - neighborhood)
    else:
        sy = 0

    if (ycell + neighborhood) > (width - 1):
        ey = (neighborhood * 2 + 1) - abs(ycell + neighborhood - width + 1)
    else:
        ey = neighborhood * 2 + 1

    trunc_distances = neighborhood_distances[sy:ey, sx:ex]

    # Compute direction based on neighborhood
    xpull = (xs * neigh_scores * trunc_distances).sum()
    ypull = (ys * neigh_scores * trunc_distances).sum()

    move_thresh = 0
    if xpull > move_thresh and xnew < (width - 1):
        xnew += 1
    elif xpull < -move_thresh and xnew > 0:
        xnew -= 1

    if ypull > move_thresh and ynew < (height - 1):
        ynew += 1
    elif ypull < -move_thresh and ynew > 0:
        ynew -= 1

    # print(xnew, ynew, xpull, ypull)
    # print(neigh_scores.max())
    # What should we do if the cell gets trapped?
    # assert(score != inf)
    return ynew, xnew


def animate_cells(
    num_free: int, regions: np.ndarray, max_iters: int, neighborhood: int
) -> Tuple[np.ndarray, List[np.ndarray]]:

    cell_grid: np.ndarray = np.zeros_like(regions, dtype=np.bool)
    height, width = cell_grid.shape

    cell_anim = np.zeros_like(regions, dtype=np.bool)
    cell_anim_steps = [cell_anim.copy()]

    # Initial cell placements (allowing colocated cells)
    cells = [(randrange(height), randrange(width), MOVING) for _ in range(num_free)]

    cell_anim = np.zeros_like(regions, dtype=np.bool)
    for (y, x, _) in cells:
        cell_anim[y, x] = True
    cell_anim_steps.append(cell_anim.copy())

    # Cells do not collide and can temporarily occupy the same space
    for iteration in range(max_iters):

        if num_free == 0:
            break

        for i, (y, x, free) in enumerate(cells):
            if free == MOVING:
                ynew, xnew = travel(y, x, regions, neighborhood, cell_grid)

                # No movement
                if ynew == y and xnew == x:
                    cell_grid[y, x] = True
                    cells[i] = (y, x, STATIONARY)
                    num_free -= 1

                # Still moving
                else:
                    cells[i] = (ynew, xnew, MOVING)

        cell_anim = np.zeros_like(regions, dtype=np.bool)
        for (y, x, _) in cells:
            cell_anim[y, x] = True
        cell_anim_steps.append(cell_anim.copy())

        if VERBOSE:
            print(f"Iteration {iteration:>3}: {num_free:>3} cells still moving.")

    return cell_grid, cell_anim_steps


def run_animations(
    regions, neighborhood, max_iterations, num_animations, num_free_cells
):

    final_grids = []
    animations = []

    for _ in range(num_animations):
        if VERBOSE:
            print()
        g, a = animate_cells(num_free_cells, regions, max_iterations, neighborhood)
        final_grids.append(g)
        animations.append(a)

    return final_grids, animations


def plot_figures(
    components, square_image, final_panes, binary_image, lattice, regions, dirname
) -> None:

    num_components = len(components)

    components_image = np.zeros_like(square_image, dtype=np.float)

    component_cmap = plt.get_cmap("Set1", num_components)

    # Shuffle to see different colorings since some colors are repeated
    shuffle(components)  # TODO: only useful in a notebook

    for i, component in enumerate(components):
        component_color = component_cmap(i / num_components)[:3]
        for (vy, vx) in component:
            components_image[vy, vx, :] = component_color

    fig, axes = plt.subplots(2, 3, figsize=(16, 16))
    axes = [ax for sublist in axes for ax in sublist]

    plots = [
        (square_image, "Original Image"),
        (binary_image, "Binarized Image"),
        (components_image, "Components Image"),
        (lattice, "Lattice Image"),
        (regions, "Scored Regions"),
        (final_panes, "After Simulation"),
    ]

    for ax, (img, ttl) in zip(axes, plots):

        if ttl == "After Simulation":
            alpha = 1 / len(final_panes)

            for g in img:
                gimage = np.zeros(list(g.shape) + [3])
                gimage[:, :, 0] = g
                ax.imshow(gimage, alpha=alpha)
        else:
            ax.imshow(img)
        ax.set_title(ttl)

    plt.savefig(path.join(dirname, "panel_image.png"))

    if VERBOSE:
        plt.tight_layout()
        plt.show()


def generate_animation_image(animation_data, dirname):
    def combine_animations(animations, frame):

        gimage = np.zeros_like(animations[0][0], dtype=int64)

        for anim in animations:
            f = frame if frame < len(anim) else -1
            gimage += anim[f] * int(alpha * 255)

        return gimage

    alpha = 1 / len(animation_data)
    fig, ax = plt.subplots()

    anim_image = np.zeros(list(animation_data[0][0].shape) + [3], dtype=np.int)
    anim_image[:, :, 0] = combine_animations(animation_data, 0)

    im = ax.imshow(anim_image)

    def animate(i):
        anim_image[:, :, 0] = combine_animations(animation_data, i)
        im.set_data(anim_image)

    max_frame = max(len(anim) for anim in animation_data)
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=max_frame)

    ani.save(path.join(dirname, "animation.gif"))

    return ani


VERBOSE = False


def main() -> None:

    import configuration as cfg

    global VERBOSE
    VERBOSE = cfg.verbose

    new_dirname = path.join(getcwd(), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    makedirs(new_dirname)

    # Copy the configuration file to the directory so that we can see
    # the values that generated the outputs
    copyfile("./configuration.py", path.join(new_dirname, "configuration.py"))

    # Force the image to be square
    # TODO: ask Ryan if we can expect most images to be about 512x512
    # TODO: better to chop?
    square_image = cfg.image.resize((512, 512))

    start = time()
    binary_image = binarize_and_threshold_image(square_image, cfg.binary_threshold)
    if VERBOSE:
        print(binary_image.shape, binary_image.dtype)
        print(f"Time to binarize the input image: {time() - start:0.3f}s\n")

    start = time()
    adj_list = create_matrix_from_adjacency_list(binary_image, cfg.connection_radius)
    if VERBOSE:
        print("Number of vertices in graph:", len(adj_list))
        print(f"Time to create graph: {time() - start:0.3f}s\n")

    start = time()
    components = find_connected_compnents(adj_list)
    num_components = len(components)
    if VERBOSE:
        print("Number of components:", num_components)
        print(f"Time to compute connected components: {time() - start:0.3f}s\n")

    start = time()
    lattice = create_lattice_of_component_counts(
        binary_image.shape,
        cfg.lattice_radius,
        cfg.lattice_stride,
        components,
        cfg.min_component_size,
    )
    if VERBOSE:
        print(f"Time to create lattice: {time() - start:0.3f}s\n")

    start = time()
    brightness = get_brightnesses_from_image(square_image, cfg.lattice_stride)
    regions = compute_region_scores(lattice, brightness)
    if VERBOSE:
        print(f"Time to create regions: {time() - start:0.3f}s\n")

    final_panes, animation_data = run_animations(
        regions,
        cfg.neighborhood,
        cfg.max_iterations,
        cfg.num_simulations,
        cfg.num_free_cells,
    )

    plot_figures(
        components,
        square_image,
        final_panes,
        binary_image,
        lattice,
        regions,
        new_dirname,
    )
    generate_animation_image(animation_data, new_dirname)


if __name__ == "__main__":
    main()
