import numpy as np
import sys
import random
import copy
import skimage.measure
import numpy.random
from skimage import morphology


# return a world, which is same as the world in PRIMAL1
def random_generator(SIZE_O=(10, 40), PROB_O=(0, .3)):
    size = (SIZE_O[0], SIZE_O[1])
    prob = (PROB_O[0], PROB_O[1])

    def primal_map(SIZE=size, PROB=prob):
        prob = np.random.triangular(PROB[0], .33 * PROB[0] + .66 * PROB[1], PROB[1])
        size = np.random.choice([SIZE[0], SIZE[0] * .5 + SIZE[1] * .5, SIZE[1]], p=[.5, .25, .25])
        # prob = self.PROB
        # size = self.SIZE  # fixed world0 size and obstacle density for evaluation
        # here is the map without any agents nor goals
        world = -(np.random.rand(int(size), int(size)) < prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id
        return world

    world = primal_map(SIZE=size, PROB=prob)
    world = np.array(world)
    return world


def maze_generator(env_size=(10, 70), wall_components=(1, 8), obstacle_density=None,
                   go_straight=0.8):
    min_size, max_size = env_size
    min_component, max_component = wall_components
    # Returns a random integer in the range [low, high)
    num_components = np.random.randint(low=min_component, high=max_component + 1)
    # the world_size must bigger than 5, while actually, the min size of the world is 10
    assert min_size > 5
    # todo: write comments
    """
    num_agents,
    IsDiagonal,
    min_size: min length of the 'radius' of the map,
    max_size: max length of the 'radius' of the map,
    complexity,
    obstacle_density,
    go_straight,
    """
    if obstacle_density is None:
        obstacle_density = [0, 1]

    def maze(h, w, total_density=0):
        # Only odd shapes
        assert h > 0 and w > 0, "You are giving non-positive width and height"
        shape = ((h // 2) * 2 + 3, (w // 2) * 2 + 3)
        # Adjust num_components and density relative to maze world_size
        # density    = int(density * ((shape[0] // 2) * (shape[1] // 2))) // 20 # world_size of components
        density = int(shape[0] * shape[1] * total_density // num_components) if num_components != 0 else 0

        # Build actual maze
        Z = np.zeros(shape, dtype='int')
        # Fill borders
        Z[0, :] = Z[-1, :] = 1  # Set the first row and the last row of Z as 1
        Z[:, 0] = Z[:, -1] = 1  # Set the first and last column of Z as 1
        # Make aisles
        for i in range(density):
            x, y = np.random.randint(0, shape[1] // 2) * 2, np.random.randint(0, shape[
                0] // 2) * 2  # pick a random position
            Z[y, x] = 1
            last_dir = 0
            for j in range(num_components):
                neighbours = []
                if x > 1:             neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:             neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    if last_dir == 0:
                        y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                        if Z[y_, x_] == 0:
                            last_dir = (y_ - y, x_ - x)
                            Z[y_, x_] = 1
                            Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
                    else:
                        index_F = -1
                        index_B = -1
                        diff = []
                        for k in range(len(neighbours)):
                            diff.append((neighbours[k][0] - y, neighbours[k][1] - x))
                            if diff[k] == last_dir:
                                index_F = k
                            elif diff[k][0] + last_dir[0] == 0 and diff[k][1] + last_dir[1] == 0:
                                index_B = k
                        assert (index_B >= 0)
                        if (index_F + 1):
                            p = (1 - go_straight) * np.ones(len(neighbours)) / (len(neighbours) - 2)
                            p[index_B] = 0
                            p[index_F] = go_straight
                            # assert(p.sum() == 1)
                        else:
                            if len(neighbours) == 1:
                                p = 1
                            else:
                                p = np.ones(len(neighbours)) / (len(neighbours) - 1)
                                p[index_B] = 0
                            assert (p.sum() == 1)

                        I = np.random.choice(range(len(neighbours)), p=p)
                        (y_, x_) = neighbours[I]
                        if Z[y_, x_] == 0:
                            last_dir = (y_ - y, x_ - x)
                            Z[y_, x_] = 1
                            Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
        return Z

    world_size = np.random.randint(min_size, max_size + 1)
    world = -maze(int(world_size), int(world_size),
                  total_density=np.random.uniform(obstacle_density[0], obstacle_density[1])).astype(int)
    world = np.array(world)
    return world

# Coding by Cheng-Yang @ May 31, 2022
def warehouse_generator(num_block, num_shelves=(10, 5), ent_location=10, ent_position='right', entrance_size=1):
    shelf_size = (2 * num_shelves[0] + num_shelves[0] - 1, 5 * num_shelves[1] + num_shelves[1] - 1)
    block_size = (shelf_size[0] + 4, shelf_size[1] + 4)
    env_size = (block_size[0] * num_block[0] + 2, block_size[1] * num_block[1] + 2)

    def warehouse(length, height):
        counter_row = 0
        counter_column = 0
        world_size = (length, height)
        # Build actual warehouse
        depot = np.zeros(world_size, dtype='int')
        # create border
        depot[0, :] = depot[-1, :] = 1  # Set the first row and the last row of Z as 1
        depot[:, 0] = depot[:, -1] = 1  # Set the first and last column of Z as 1
        for i in range(1, num_block[0] * num_shelves[0] + 1):
            for j in range(1, num_block[1] * num_shelves[1] + 1):
                counter_row = int((i - 1) // num_shelves[0])
                counter_column = int((j - 1) // num_shelves[1])
                depot[3 * i + counter_row * 3:3 * i + counter_row * 3 + 2,
                3 + 6 * (j - 1) + counter_column * 3:3 + 6 * (j - 1) + 5 + counter_column * 3] = 1
                # depot[3*i:3*i+2, 3+6*(j-1):3+6*(j-1)+5] = 1
        return depot

    grid_map = -warehouse(int(env_size[0]), int(env_size[1])).astype(int)
    for i in range(0, entrance_size):
        loc = ent_location + i
        if ent_position == "top":
            if loc >= env_size[1]:
                loc = env_size[1] - 1
            entrance_location = (0, loc)
        elif ent_position == "bottom":
            if loc >= env_size[1]:
                loc = env_size[1] - 1
            entrance_location = (env_size[0] - 1, loc)
        elif ent_position == "left":
            if loc >= env_size[0]:
                loc = env_size[0] - 1
            entrance_location = (loc, 0)
        else:
            if loc >= env_size[0]:
                loc = env_size[0] - 1
            entrance_location = (loc, env_size[1] - 1)

        grid_map[entrance_location[0], entrance_location[1]] = 0

    grid_map = np.array(grid_map)
    return grid_map


def get_map_nodes(world):
    """
    this function should be called per episode
    """
    def neighbour(x, y, image):
        """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
        '''This function will work only if image[i, j] == 0'''
        num_free_cell = 0
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
        if x_1 >= 0 and y_1 >= 0 and x1 < image.shape[0] and y1 < image.shape[1]:
            if image[x, y] == 0:
                for i in range(x_1, x1 + 1):
                    for j in range(y_1, y1 + 1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        elif x == 0 and y != 0 and y != image.shape[1] - 1:
            if image[x, y] == 0:
                for i in range(x, x1 + 1):
                    for j in range(y_1, y1 + 1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        elif x == image.shape[0] - 1 and y != 0 and y != image.shape[1] - 1:
            if image[x, y] == 0:
                for i in range(x_1, x1):
                    for j in range(y_1, y1 + 1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        elif x != 0 and x != image.shape[0] - 1 and y == 0:
            if image[x, y] == 0:
                for i in range(x_1, x1 + 1):
                    for j in range(y, y1 + 1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        elif x != 0 and x != image.shape[0] - 1 and y == image.shape[1] - 1:
            if image[x, y] == 0:
                for i in range(x_1, x1 + 1):
                    for j in range(y_1, y1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        elif x == 0 and y == 0:
            if image[x, y] == 0:
                for i in range(x, x1 + 1):
                    for j in range(y, y1 + 1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        elif x == 0 and y == image.shape[1] - 1:
            if image[x, y] == 0:
                for i in range(x, x1 + 1):
                    for j in range(y_1, y1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        elif x == image.shape[0] - 1 and y == 0:
            if image[x, y] == 0:
                for i in range(x_1, x1):
                    for j in range(y, y1 + 1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        elif x == image.shape[0] - 1 and y == image.shape[1] - 1:
            if image[x, y] == 0:
                for i in range(x_1, x1):
                    for j in range(y_1, y1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        # we do this bc we have remove the ego free cell
        num_free_cell = num_free_cell - 1
        return num_free_cell

    def end_branch_point(image):
        ed_points = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                num_neighbours = neighbour(i, j, image)
                if image[i, j] == 0:
                    if num_neighbours == 1 or num_neighbours >= 3:
                        ed_points.append([i, j])
        return ed_points

    def mask_ebpoints(image, eb_points):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if i == 0 and j == 0:
                    if ([i, j] in eb_points) and ([i, j + 1] in eb_points) and ([i + 1, j] in eb_points):
                        eb_points.remove([i, j + 1])
                        eb_points.remove([i + 1, j])
                elif i == image.shape[0] - 1 and j == 0:
                    if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j + 1] in eb_points):
                        eb_points.remove([i - 1, j])
                        eb_points.remove([i, j + 1])
                elif i == 0 and j == image.shape[1] - 1:
                    if ([i, j] in eb_points) and ([i, j - 1] in eb_points) and ([i + 1, j] in eb_points):
                        eb_points.remove([i, j - 1])
                        eb_points.remove([i + 1, j])
                elif i == image.shape[0] - 1 and j == image.shape[1] - 1:
                    if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j - 1] in eb_points):
                        eb_points.remove([i - 1, j])
                        eb_points.remove([i, j - 1])
                elif i == 0 and j != 0 and j != image.shape[1] - 1:
                    if ([i, j] in eb_points) and ([i, j + 1] in eb_points) and ([i + 1, j] in eb_points):
                        eb_points.remove([i, j + 1])
                        eb_points.remove([i + 1, j])
                    if ([i, j] in eb_points) and ([i, j - 1] in eb_points) and ([i + 1, j] in eb_points):
                        eb_points.remove([i, j - 1])
                        eb_points.remove([i + 1, j])
                elif i == image.shape[0] - 1 and j != 0 and j != image.shape[1] - 1:
                    if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j + 1] in eb_points):
                        eb_points.remove([i - 1, j])
                        eb_points.remove([i, j + 1])
                    if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j - 1] in eb_points):
                        eb_points.remove([i - 1, j])
                        eb_points.remove([i, j - 1])
                elif i != 0 and i != image.shape[0] - 1 and j == 0:
                    if ([i, j] in eb_points) and ([i, j + 1] in eb_points) and ([i + 1, j] in eb_points):
                        eb_points.remove([i, j + 1])
                        eb_points.remove([i + 1, j])
                    if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j + 1] in eb_points):
                        eb_points.remove([i - 1, j])
                        eb_points.remove([i, j + 1])
                elif i != 0 and i != image.shape[0] - 1 and j == image.shape[1] - 1:
                    if ([i, j] in eb_points) and ([i, j - 1] in eb_points) and ([i + 1, j] in eb_points):
                        eb_points.remove([i, j - 1])
                        eb_points.remove([i + 1, j])
                    if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j - 1] in eb_points):
                        eb_points.remove([i - 1, j])
                        eb_points.remove([i, j - 1])
                else:
                    if ([i, j] in eb_points) and ([i, j + 1] in eb_points) and ([i + 1, j] in eb_points):
                        eb_points.remove([i, j + 1])
                        eb_points.remove([i + 1, j])
                    if ([i, j] in eb_points) and ([i, j - 1] in eb_points) and ([i + 1, j] in eb_points):
                        eb_points.remove([i, j - 1])
                        eb_points.remove([i + 1, j])
                    if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j + 1] in eb_points):
                        eb_points.remove([i - 1, j])
                        eb_points.remove([i, j + 1])
                    if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j - 1] in eb_points):
                        eb_points.remove([i - 1, j])
                        eb_points.remove([i, j - 1])
        return eb_points

    world_for_ske = 1 - (-1 * world)
    skeleton, distance = morphology.medial_axis(world_for_ske, return_distance=True)
    ske_needed = skeleton.astype(int) - 1
    eb_points = end_branch_point(ske_needed)
    nodes = mask_ebpoints(ske_needed, eb_points)
    return nodes


def house_generator(env_size=10, obstacle_ratio=10, remove_edge_ratio=6):
    world_size = env_size
    world = np.zeros((world_size, world_size))
    all_x = range(2, world_size - 2)
    all_y = range(2, world_size - 2)
    obs_edge = []
    obs_corner_x = []
    while len(obs_corner_x) < world_size // obstacle_ratio:
        corn_x = random.sample(all_x, 1)
        near_flag = False
        for i in obs_corner_x:
            if abs(i - corn_x[0]) == 1:
                near_flag = True
        if not near_flag:
            obs_corner_x.append(corn_x[0])
    obs_corner_y = []
    while len(obs_corner_y) < world_size // obstacle_ratio:
        corn_y = random.sample(all_y, 1)
        near_flag = False
        for i in obs_corner_y:
            if abs(i - corn_y[0]) == 1:
                near_flag = True
        if not near_flag:
            obs_corner_y.append(corn_y[0])
    obs_corner_x.append(0)
    obs_corner_x.append(world_size - 1)
    obs_corner_y.append(0)
    obs_corner_y.append(world_size - 1)

    for i in obs_corner_x:
        edge = []
        for j in range(world_size):
            world[i][j] = 1
            if j not in obs_corner_y:
                edge.append([i, j])
            if j in obs_corner_y and edge != []:
                obs_edge.append(edge)
                edge = []

    for i in obs_corner_y:
        edge = []
        for j in range(world_size):
            world[j][i] = 1
            if j not in obs_corner_x:
                edge.append([j, i])
            if j in obs_corner_x and edge != []:
                obs_edge.append(edge)
                edge = []

    all_edge_list = range(len(obs_edge))
    remove_edge = random.sample(all_edge_list, len(obs_edge) // remove_edge_ratio)
    for edge_number in remove_edge:
        for current_edge in obs_edge[edge_number]:
            world[current_edge[0]][current_edge[1]] = 0

    for edges in obs_edge:
        if len(edges) == 1 or len(edges) <= world_size // 20:
            for coordinates in edges:
                world[coordinates[0]][coordinates[1]] = 0
    _, count = skimage.measure.label(world, background=1, connectivity=1, return_num=True)

    while count != 1 and len(obs_edge) > 0:
        door_edge_index = random.sample(range(len(obs_edge)), 1)[0]
        door_edge = obs_edge[door_edge_index]
        door_index = random.sample(range(len(door_edge)), 1)[0]
        door = door_edge[door_index]
        world[door[0]][door[1]] = 0
        _, count = skimage.measure.label(world, background=1, connectivity=1, return_num=True)
        obs_edge.remove(door_edge)

    world[:, -1] = world[:, 0] = 1
    world[-1, :] = world[0, :] = 1

    nodes_obs = get_map_nodes(world)

    return world, nodes_obs


def partition(grid_world):
    obs_world, obs_count = skimage.measure.label(grid_world, background=0, connectivity=2, return_num=True)
    return obs_world, obs_count


def neighbour(x, y, image):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    '''This function will work only if image[i, j] == 0'''
    num_free_cell = 0
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    if x_1 >= 0 and y_1 >= 0 and x1 < image.shape[0] and y1 < image.shape[1]:
        if image[x, y] == 0:
            for i in range(x_1, x1 + 1):
                for j in range(y_1, y1 + 1):
                    if image[i, j] == 0:
                        num_free_cell = num_free_cell + 1
    elif x == 0 and y != 0 and y != image.shape[1] - 1:
        if image[x, y] == 0:
            for i in range(x, x1 + 1):
                for j in range(y_1, y1 + 1):
                    if image[i, j] == 0:
                        num_free_cell = num_free_cell + 1
    elif x == image.shape[0] - 1 and y != 0 and y != image.shape[1] - 1:
        if image[x, y] == 0:
            for i in range(x_1, x1):
                for j in range(y_1, y1 + 1):
                    if image[i, j] == 0:
                        num_free_cell = num_free_cell + 1
    elif x != 0 and x != image.shape[0] - 1 and y == 0:
        if image[x, y] == 0:
            for i in range(x_1, x1 + 1):
                for j in range(y, y1 + 1):
                    if image[i, j] == 0:
                        num_free_cell = num_free_cell + 1
    elif x != 0 and x != image.shape[0] - 1 and y == image.shape[1] - 1:
        if image[x, y] == 0:
            for i in range(x_1, x1 + 1):
                for j in range(y_1, y1):
                    if image[i, j] == 0:
                        num_free_cell = num_free_cell + 1
    elif x == 0 and y == 0:
        if image[x, y] == 0:
            for i in range(x, x1 + 1):
                for j in range(y, y1 + 1):
                    if image[i, j] == 0:
                        num_free_cell = num_free_cell + 1
    elif x == 0 and y == image.shape[1] - 1:
        if image[x, y] == 0:
            for i in range(x, x1 + 1):
                for j in range(y_1, y1):
                    if image[i, j] == 0:
                        num_free_cell = num_free_cell + 1
    elif x == image.shape[0] - 1 and y == 0:
        if image[x, y] == 0:
            for i in range(x_1, x1):
                for j in range(y, y1 + 1):
                    if image[i, j] == 0:
                        num_free_cell = num_free_cell + 1
    elif x == image.shape[0] - 1 and y == image.shape[1] - 1:
        if image[x, y] == 0:
            for i in range(x_1, x1):
                for j in range(y_1, y1):
                    if image[i, j] == 0:
                        num_free_cell = num_free_cell + 1
    # we do this bc we have remove the ego free cell
    num_free_cell = num_free_cell - 1
    return num_free_cell


def end_branch_point(image):
    ed_points = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            num_neighbours = neighbour(i, j, image)
            if image[i, j] == 0:
                if num_neighbours == 1 or num_neighbours >= 3:
                    ed_points.append([i, j])
    return ed_points


def mask_ebpoints(image, eb_points):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i == 0 and j == 0:
                if ([i, j] in eb_points) and ([i, j + 1] in eb_points) and ([i + 1, j] in eb_points):
                    eb_points.remove([i, j + 1])
                    eb_points.remove([i + 1, j])
            elif i == image.shape[0] - 1 and j == 0:
                if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j + 1] in eb_points):
                    eb_points.remove([i - 1, j])
                    eb_points.remove([i, j + 1])
            elif i == 0 and j == image.shape[1] - 1:
                if ([i, j] in eb_points) and ([i, j - 1] in eb_points) and ([i + 1, j] in eb_points):
                    eb_points.remove([i, j - 1])
                    eb_points.remove([i + 1, j])
            elif i == image.shape[0] - 1 and j == image.shape[1] - 1:
                if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j - 1] in eb_points):
                    eb_points.remove([i - 1, j])
                    eb_points.remove([i, j - 1])
            elif i == 0 and j != 0 and j != image.shape[1] - 1:
                if ([i, j] in eb_points) and ([i, j + 1] in eb_points) and ([i + 1, j] in eb_points):
                    eb_points.remove([i, j + 1])
                    eb_points.remove([i + 1, j])
                if ([i, j] in eb_points) and ([i, j - 1] in eb_points) and ([i + 1, j] in eb_points):
                    eb_points.remove([i, j - 1])
                    eb_points.remove([i + 1, j])
            elif i == image.shape[0] - 1 and j != 0 and j != image.shape[1] - 1:
                if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j + 1] in eb_points):
                    eb_points.remove([i - 1, j])
                    eb_points.remove([i, j + 1])
                if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j - 1] in eb_points):
                    eb_points.remove([i - 1, j])
                    eb_points.remove([i, j - 1])
            elif i != 0 and i != image.shape[0] - 1 and j == 0:
                if ([i, j] in eb_points) and ([i, j + 1] in eb_points) and ([i + 1, j] in eb_points):
                    eb_points.remove([i, j + 1])
                    eb_points.remove([i + 1, j])
                if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j + 1] in eb_points):
                    eb_points.remove([i - 1, j])
                    eb_points.remove([i, j + 1])
            elif i != 0 and i != image.shape[0] - 1 and j == image.shape[1] - 1:
                if ([i, j] in eb_points) and ([i, j - 1] in eb_points) and ([i + 1, j] in eb_points):
                    eb_points.remove([i, j - 1])
                    eb_points.remove([i + 1, j])
                if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j - 1] in eb_points):
                    eb_points.remove([i - 1, j])
                    eb_points.remove([i, j - 1])
            else:
                if ([i, j] in eb_points) and ([i, j + 1] in eb_points) and ([i + 1, j] in eb_points):
                    eb_points.remove([i, j + 1])
                    eb_points.remove([i + 1, j])
                if ([i, j] in eb_points) and ([i, j - 1] in eb_points) and ([i + 1, j] in eb_points):
                    eb_points.remove([i, j - 1])
                    eb_points.remove([i + 1, j])
                if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j + 1] in eb_points):
                    eb_points.remove([i - 1, j])
                    eb_points.remove([i, j + 1])
                if ([i, j] in eb_points) and ([i - 1, j] in eb_points) and ([i, j - 1] in eb_points):
                    eb_points.remove([i - 1, j])
                    eb_points.remove([i, j - 1])
    return eb_points


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    print("PRIMAL1 MAP")
    # bc of the need to display multiple images, the display mode of matplotlib is switched to interactive mode.
    # The code continues to execute even if plt.show() is encountered in the script
    plt.ion()
    # repeat the cycle 1000 times， i.e., generate 1000 ENVs
    for _ in range(1):
        # create the world/env (maze type)
        # generator = maze_generator()
        # create a warehouse
        # world = random_generator(SIZE_O=(38, 40), PROB_O=(0, .3))
        # world = warehouse_generator(num_block=(2, 2), num_shelves=(4, 2), ent_location=10, ent_position="top", entrance_size=3)
        # world = maze_generator(env_size=(39, 40), wall_components=(7, 8), obstacle_density=(0, .3))
        world, nodes = house_generator(env_size=(10, 40))
        world_for_ske = 1 - (-1 * world)
        # skeleton = morphology.skeletonize(world_for_ske)
        skeleton, distance = morphology.medial_axis(world_for_ske, return_distance=True)
        ske_needed = skeleton.astype(int) - 1
        eb_points = end_branch_point(ske_needed)
        world1 = copy.deepcopy(world)
        world2 = copy.deepcopy(ske_needed)
        for i in eb_points:
            world1[i[0]][i[1]] = 1
            world2[i[0]][i[1]] = 1
        masked_eb_points = mask_ebpoints(ske_needed, eb_points)
        # ske_neededs = skeletons.astype(int) - 1
        # print(f"{np.size(world, 1)}")
        # obs_p, obs_n = partition(world)
        world3 = copy.deepcopy(world)
        world4 = copy.deepcopy(ske_needed)
        for i in masked_eb_points:
            world3[i[0]][i[1]] = 1
            world4[i[0]][i[1]] = 1
        count_nodes = 0
        for i in range(world3.shape[0]):
            for j in range(world3.shape[0]):
                if world3[i, j] == 1:
                    count_nodes += 1
        print(count_nodes)

        # node_pairs = []
        # is_neibor = True
        # for i in masked_eb_points:
        #     for j in masked_eb_points:
        #         if i != j:
        #             i_tuple = tuple(i)
        #             j_tuple = tuple(j)
        #             path, _ = astar_8(ske_needed, i_tuple, j_tuple)
        #             for ns in path:
        #                 nodes_list = list(ns)
        #                 if (nodes_list in masked_eb_points) and (nodes_list != i) and (ns != j):
        #                     is_neibor = False
        #                     break
        #             if is_neibor:
        #                 node_pairs.append([i_tuple, j_tuple])
        #             is_neibor = True
        # print(node_pairs)
        # edge = []
        # edges = dict()
        # for i in masked_eb_points:
        #     i = tuple(i)
        #     node_str = str(i)
        #     for j in range(len(node_pairs)):
        #         if i == node_pairs[j][0]:
        #             edge.append(node_pairs[j][1])
        #     edges.update({node_str: edge})
        #     edge = []
        # print(edges)

        # i = [3, 4]
        # path_length = []
        # node_sequence = []
        # for j in masked_eb_points:
        #     i_tuple = tuple(i)
        #     j_tuple = tuple(j)
        #     try:
        #         path, _ = astar_4(world, i_tuple, j_tuple)
        #     except TypeError:
        #         print(world)
        #         print('i_tuple', i_tuple)
        #         print('j_tuple', j_tuple)
        #     path_length.append(len(path))
        #     node_sequence.append(j)
        # path_length, node_sequence = (list(t) for t in zip(*sorted(zip(path_length, node_sequence))))
        # nearest_node = (node_sequence[:1])
        # print('[3, 4]s nearest node: ', nearest_node)

        # node, edge, all_path = generate_graph(world[0], ske_needed, obs_n * 1.5, neighbors=3)
        # for i in all_path:
        #     for j in i:
        #         if j != i:
        #             world1[j[0]][j[1]] = 1
        # for i in node:
        #     world1[i[0]][i[1]] = 3
        plt.imshow(world)
        plt.pause(0.1)
        plt.imshow(ske_needed)
        # plt.imshow(labeled_img)
        plt.pause(0.1)
        plt.imshow(world1)
        plt.pause(0.1)
        plt.imshow(world2)
        plt.pause(0.1)
        plt.imshow(world3)
        plt.pause(0.1)
        plt.imshow(world4)
        plt.pause(0.1)
        # plt.imshow(padding_world)
        # turn off the interactive mode before show the figures
    plt.ioff()
    plt.show()
