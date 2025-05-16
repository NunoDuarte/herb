import numpy as np


def get_min_box(heightmap, bin):
    # function that calculates the minimum box that contains the object
    length_resolution = bin[0] / heightmap.shape[0]
    width_resolution = bin[1] / heightmap.shape[1]

    # get the indices of the non-zero values
    indices = np.nonzero(heightmap)

    # check if the object is empty
    if len(indices[0]) == 0:
        return 0.0
    
    # get the minimum and maximum values of the indices
    min_x = np.min(indices[0])
    max_x = np.max(indices[0])
    min_y = np.min(indices[1])
    max_y = np.max(indices[1])
    # calculate the length and width of the object
    length = (max_x - min_x) * length_resolution
    width = (max_y - min_y) * width_resolution
    # get the height of the object
    height = np.max(heightmap)
    # calculate the volume of the object
    min_box_volume = length * width * height
    
    return min_box_volume


if __name__ == '__main__':
    exit(0)