import numpy as np
from stable_baselines3.sac.policies import CnnPolicy
import cv2
import matplotlib.pyplot as plt
import os

def print_heighmap_norm(heightMap, vmin = 0, vmax = 255, save = False, savename='test.png', show = True):
    # close previous figure
    plt.close()
    # draw new figure
    plt.figure()
    plt.imshow(heightMap,  cmap=plt.cm.hot, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if save:
        plt.savefig(savename)
    if show:
        plt.title(savename)
        plt.show()

# unnormalize from [-1 to 1]
def unnormalize(x, min, max):
    return 0.5 * (x + 1) * (max - min) + min

def prepare_heightmap(image, camera_offset, eps=0.05):
    """
    takes an image and returns a 1x224x224 compatible image
    """

    # (1) remove the camera offset so the table is at 0
    image = image - camera_offset
    image = image[5:, 10:]
    # (1a) possibly filter if it does not look ok with some threshold
    image[image < eps] = 0

    # (2) make sure the image is 115x76, if not resize it
    if image.shape != (115, 76):
        image = cv2.resize(image, (76, 115))
        
    image = np.flip(image)
    
    # (3) if necessary mask some rows or columns with 0 to remove the sides of the box

    # (4) standard processing from the environment
    # scale the heightmap from [0, 0.3] to [0, 255]
    heightmap = (image / 0.3) * 255

    heightmap = np.expand_dims(heightmap, axis=0)

    return heightmap

model_pth = 'policies/0221_for_human_comparison.pkl'
policy = CnnPolicy.load(model_pth)
policy.eval()
print('Model loaded')

path = 'environment/physics0/images/'

# order the files alphabetically
filenames = sorted([
    f for f in os.listdir(path)
])

search_string = '_p88_s1_24'
for file in filenames:
    if search_string in file:
        right_files = os.path.join(path, file)
        print(right_files)

        image = np.load(right_files)
        image = image / 1000.0
        # print(np.min(image))
        # print(np.max(image))

        obs = prepare_heightmap(image, camera_offset=0.0)

        print_heighmap_norm(obs[0], savename=right_files)

