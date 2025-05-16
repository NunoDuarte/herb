import gymnasium as gym
import numpy as np
from simulator.Interface import Interface
from simulator.space import Space, draw_heatmap
from collections import Counter
import pickle
from utils.volume_utils import get_min_box
from scipy.spatial.transform import Rotation as R
from simulator.space import draw_heatmap_norm
import time
from simulator.tools import extendMat
import transforms3d
import json
import pandas as pd
import time
import trimesh



#from Andre's Unity code
BOX_VOLUME = 0.0140426

ALL_OBJECTS = [
    "002 MasterChef Can",
    "003 Cracker Box",
    "004 Sugar Box",
    "005 Tomato Coup Can",
    "006 Mustard Bottle",
    "007 Tuna Fish Can",
    "008 Pudding Box",
    "010 Potted Meat Can",
    "011 Banana",
    "012 Strawberry",
    "013 Apple",
    "014 Lemon",
    "015 Peach",
    "016 Pear",
    "017 Orange",
    "018 Plum",
    "021 Bleach Cleanser",
    "025 Mug",
    "057 Racquetball",
    "058 Golf Ball",
    "100 Half Egg Carton",
    "101 Bread",
    "102 toothbrush",
    "103 toothpaste",
]

# get the transition probabilities
ALL_TRANSITIONS = pickle.load(open('environment/physics0/all_transitions.pkl', 'rb'))

# normalize from [-1 to 1]
def normalize(x, min, max):
    return (2 * (x - min) / (max - min)) - 1

# unnormalize from [-1 to 1]
def unnormalize(x, min, max):
    return 0.5 * (x + 1) * (max - min) + min

class PackingGameHeuristic(gym.Env):
    def __init__(self, bin_size=[0.345987, 0.227554, 0.1637639], object_info='dataset/datas/object_info.npz', visual=False, ordered_objs='largest', reward_function='simple', alpha=0.9, 
                 unpacked_list_min=0.7, unpacked_list_max=0.9, move_res = 25, rotation_bins=24):
        super().__init__()
        self.bin = np.array(bin_size)
        self.ordered_objs = ordered_objs
        # self.bin = np.array([0.355987, 0.237554, 0.1637639]) # add a cm to the sides
        self.interface = Interface(visual=visual, bin=self.bin)
        self.space = Space(self.bin, resolutionAct=0.003, resolutionH=0.003, ZRotNum=8, scale=[1.0, 1.0, 1.0], cut_last_row=True)

        self.unpacked_list_min = unpacked_list_min
        self.unpacked_list_max = unpacked_list_max
        
        self.observation_space = gym.spaces.Box(low=0, high=255,shape=(1, 224, 224), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        self.move_res = move_res
        # do pose bins by discretizing bin[0] and bin[1] into grid
        x_pixels = np.linspace(0, 115, move_res).astype(int)
        y_pixels = np.linspace(0, 76, move_res).astype(int)
        self.pixel_bins = np.array(np.meshgrid(x_pixels, y_pixels)).T.reshape(-1, 2)
        self.pose_bins = self.pixel_bins * self.space.resolutionH


        self.rotation_bins = rotation_bins
        self.rotation_bins = np.linspace(0, 180.0, self.rotation_bins)
        self.rotation_bins = np.round(self.rotation_bins, 3)


        if reward_function == 'simple':
            self.reward_function = self.simple_reward
        elif reward_function == 'compactness':
            self.reward_function = self.compactness_reward
        elif reward_function == 'compactness_stability':
            self.reward_function = self.compactness_stability_reawrd
            self.alpha = alpha
        else:
            raise ValueError('reward_function not implemented')
        
        # Load the object_info.npz file
        object_info = np.load(object_info)
        self.objects = object_info['objects']
        self.volumes = object_info['volumes']
        self.bbox_volumes = object_info['bbox_volumes']
        self.projections = object_info['projections']

        # spawn all objects at -100, -100, -100 to add the to the shapeMap
        for obj in self.objects:
            print(obj)
            self.interface.addObjectC(name=obj, targetC=[-100, -100, -100], rotation=[0, 0, 0], scale=[1.0, 1.0, 1.0])
        self.interface.reset()
        print('PackingGame initialized')

    def reset(self, seed=None, options=None, unpacked_list=None):
        # print('reset')
        self.space.reset()
        self.interface.reset()

        if unpacked_list is not None:
            self.unpacked_list = unpacked_list
        else:
            self.unpacked_list = self.generate_unpacked_list()

        if self.ordered_objs == 'largest':
            self.unpacked_list = self.order_objects(self.unpacked_list)

        self.cummulative_volume = 0.0

        # debugging
        # self.unpacked_list = self.unpacked_list[:3]

        self.space.shot_whole()
        heightmap = self.space.heightmapC
        observation = self.prepare_observation_heuristic(heightmap)

        return observation, {}
    
    def reset_with_time(self, seed=None, options=None, unpacked_list=None):
        # print('reset')
        self.space.reset()
        self.interface.reset()

        if unpacked_list is not None:
            self.unpacked_list = unpacked_list
        else:
            self.unpacked_list = self.generate_unpacked_list()

        if self.ordered_objs:
            self.unpacked_list = self.order_objects(self.unpacked_list)

        self.cummulative_volume = 0.0

        # debugging
        # self.unpacked_list = self.unpacked_list[:3]

        start = time.time()

        self.space.shot_whole()
        heightmap = self.space.heightmapC
        observation = self.prepare_observation(heightmap, self.projections[self.objects.tolist().index(self.unpacked_list[0])])

        elapsed_time = time.time() - start
        
        return observation, {}, elapsed_time

    def step(self, targetC, rotation, objname=None):
        # print('rotation:', rotation)
        # throw error if objname is None
        if objname is None:
            objname = self.unpacked_list[0]
            if objname is None:
                raise ValueError('objname is None')
        
        # self.interface.addObjectC(name=objname, targetC=targetC, rotation=rotation, scale=[1.0, 1.0, 1.0])
        self.interface.addObject(name=objname, targetFLB=targetC, rotation=rotation, scale=[1.0, 1.0, 1.0])
        self.interface.simulateToQuasistatic(linearTol = 0.01, angularTol = 0.01, batch = 1.0, dt = 0.01, maxBatch = 2)
        self.interface.disableObject(self.interface.objs[-1])

        # remove the placed object
        self.unpacked_list.pop(0)

        # check is the object is inside the bin
        obj_inside = self.interface.check_obj_inside(self.interface.objs[-1], checkZ=0.13)

        self.current_obj_volume = self.volumes[self.objects.tolist().index(objname)]

        if obj_inside:
            self.cummulative_volume += self.current_obj_volume

        self.space.shot_whole()
        heightmap = self.space.heightmapC

       
        # next objname
        if len(self.unpacked_list) > 0:
            next_objname = self.unpacked_list[0]
        else:
            next_objname = objname

        # prepare the observation
        observation = self.prepare_observation_heuristic(heightmap)

        # calculate the reward
        reward, done = self.reward_function(obj_inside)
        

        return observation, reward, done, False, {}
    
    def step_with_metrics(self, targetFLB, rotation, objname=None):

        # throw error if objname is None
        if objname is None:
            objname = self.unpacked_list[0]
            if objname is None:
                raise ValueError('objname is None')
        
        # self.interface.addObjectC(name=objname, targetC=targetC, rotation=rotation, scale=[1.0, 1.0, 1.0])
        self.interface.addObject(name=objname, targetFLB=targetFLB, rotation=rotation, scale=[1.0, 1.0, 1.0])
        self.interface.simulateToQuasistatic(linearTol = 0.01, angularTol = 0.01, batch = 1.0, dt = 0.01, maxBatch = 2)
        self.interface.disableObject(self.interface.objs[-1])

        # remove the placed object
        self.unpacked_list.pop(0)

        # check is the object is inside the bin
        obj_inside = self.interface.check_obj_inside(self.interface.objs[-1], checkZ=0.13)

        
        self.current_obj_volume = self.volumes[self.objects.tolist().index(objname)]

        
        self.space.shot_whole()
        heightmap = self.space.heightmapC

       
        # next objname
        if len(self.unpacked_list) > 0:
            next_objname = self.unpacked_list[0]
        else:
            next_objname = objname
        
        # prepare the observation
        observation = self.prepare_observation_heuristic(heightmap)

        success = 0

        if obj_inside:
            min_box_volume = get_min_box(heightmap, self.bin)
            
            if min_box_volume == 0:
                success = -1
                done = True
                return observation, done, success, 0, 0

            self.cummulative_volume += self.current_obj_volume
            compactness = self.cummulative_volume / get_min_box(heightmap, self.bin)

            obj = self.interface.objs[-1]
            rotation = self.interface.get_Wraped_Position_And_Orientation(obj)[1]
            # convert quaternion to Euler
            r = R.from_quat(rotation)
            rotation = r.as_euler('xyz', degrees=True)
            if (np.abs(rotation[0]) > 10 and np.abs(rotation[0]) < 170) or (np.abs(rotation[1]) > 10 and np.abs(rotation[1]) < 170):
                stability = 0
            else:
                stability = 1
            
            if len(self.unpacked_list) == 0:
                done = True
                success = 1
            else:
                done = False
            
            return observation, done, success, compactness, stability
        
        else:
            success = -1
            done = True
            return observation, done, success, 0, 0
    
    def replay_step(self, pose, objname):
        # to be used to replay the recorded poses
        targetC = pose[:3]
        rotation = pose[3:]
        PoseT = (targetC, rotation)

        mesh, _, _ = self.interface.shapeMap[objname]

        self.interface.addObjectC(name=objname, targetC=targetC, rotation=rotation, scale=[1.0, 1.0, 1.0])
        self.interface.simulateToQuasistatic(linearTol = 0.01, angularTol = 0.01, batch = 1.0, dt = 0.01, maxBatch = 2)
        self.interface.disableObject(self.interface.objs[-1])

        # check is the object is inside the bin
        obj_inside = self.interface.check_obj_inside(self.interface.objs[-1], checkZ=0.13)

        self.current_obj_volume = self.volumes[self.objects.tolist().index(objname)]

        # remove the placed object
        self.unpacked_list.pop(0)

         # next objname
        if len(self.unpacked_list) > 0:
            next_objname = self.unpacked_list[0]
        else:
            next_objname = objname

        if obj_inside:
            self.cummulative_volume += self.current_obj_volume

        self.space.shot_whole()
        heightmap = self.space.heightmapC

        # prepare the observation
        observation = self.prepare_observation(heightmap, self.projections[self.objects.tolist().index(next_objname)])

        

        # calculate the reward
        reward, done = self.reward_function(obj_inside)

        return observation, reward, done, False, {}


        

    def render(self, heightname='height.png', colorname='color.png'):
        self.space.shot_whole()
        draw_heatmap(self.space.heightmapC, save=True, savename=heightname, show=False)
        self.interface.saveImage(colorname)

    def close(self):
        self.interface.close()
        self.space.close()

    def prepare_observation_heuristic(self, heightmap):
        heightmap = (heightmap / 0.3) * 255
        heightmap = heightmap.astype(np.uint8)
        heightmap = np.expand_dims(heightmap, axis=0)
        return heightmap

    def prepare_observation(self, heightmap, projection):
        # scale the heightmap from [0, 0.3] to [0, 255]
        heightmap = (heightmap / 0.3) * 255
        
        # pad the heightmap from 115x76 to 224x224
        padded_heightmap = np.zeros((224, 224))
        padded_heightmap[54:54+heightmap.shape[0], 26:26+heightmap.shape[1]] = heightmap

        # add the projection to the padded heightmap
        padded_heightmap[76:76+projection.shape[0], 100+26:(100+26+projection.shape[1])] = projection

        # convert the padded heightmap to uint8
        padded_heightmap = padded_heightmap.astype(np.uint8)

        # convert the padded heightmap to [1 x 224 x 224]
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)



        return padded_heightmap

    # Adapted from Andre's code
    def generate_unpacked_list(self):
        
        min_volume = self.unpacked_list_min
        max_volume = self.unpacked_list_max
        
        targetVolumeRatio = np.random.uniform(min_volume, max_volume)
        curVolume = 0.0
        prevVolume = 0.0

        subset = []

        while(curVolume < targetVolumeRatio * BOX_VOLUME):
            obj = np.random.choice(ALL_OBJECTS)
            subset.append(obj)
            prevVolume = curVolume
            # add the bbox volume of the object to the current volume
            curVolume += self.bbox_volumes[ALL_OBJECTS.index(obj)]
        
        if ((curVolume > max_volume * BOX_VOLUME) or (abs(targetVolumeRatio * BOX_VOLUME - curVolume) > abs(targetVolumeRatio * BOX_VOLUME - prevVolume))):
            subset.pop()

        return subset

    def beam_search(self, unpacked_objs, all_transitions, beam_width, max_len=None):
        """
        Beam search over set of objects unpacked_objs, according to probabilities in markov_chain.
        :param beam_width: Maximum beam width during search. Actual beam width may be smaller if there aren't enough valid
        child nodes.
        :param unpacked_objs: Objects that were not yet packed.
        :param all_transitions: All transition probabilities obtained from dataset.
        :param max_len: (optional) Maximum number of elements in a predicted sequence.
        :return: Sequence of objects to be packed, in order.
        """
        min_objs = min(3, len(unpacked_objs))  # Minimum number of objects that a predicted sequence can have
        unpacked_objs = [ALL_OBJECTS.index(x) + 1 for x in
                        unpacked_objs]  # Indices of objects that were not packed yet

        open_list = [([0], float(1))]  # Stores all search beams (sequence and corresponding probability)
        while True:
            has_next_level = True
            level_counter = 0
            while has_next_level:
                has_next_level = False
                has_valid_objs = False  # Flags that, for this transition level, at least one beam has valid next objects
                new_open_list = []  # For each transition level, stores the beams that have valid next objects.
                for beam in open_list:

                    # If this beam has reached maximum length, don't add any more objects to it
                    if max_len and len(beam[0])-1 == max_len:
                        continue

                    # Check if the last obj in this beam has another transition level
                    if level_counter+1 < len(all_transitions[beam[0][-1]]):
                        has_next_level = True
                    # If this beam has no transition level for level level_counter, remove it since other beams have
                    if level_counter >= len(all_transitions[beam[0][-1]]):
                        open_list.remove(beam)
                        continue
                    # Search the current transition level for valid objects
                    valid_next_objs = []
                    valid_next_objs = valid_next_objs + \
                                    [x for x in all_transitions[beam[0][-1]][level_counter] if
                                    (x[0] in list((Counter(unpacked_objs) - Counter(beam[0])).elements()))]

                    # If there are valid next objects, branch out the beams with them and remove original beam
                    if valid_next_objs:
                        new_open_list += [(beam[0] + [x[0]], beam[1] * x[1]) for x in valid_next_objs]
                        open_list.remove(beam)
                        has_valid_objs = True

                # If at least one beam received a new object, continue the search
                if has_valid_objs:
                    open_list.clear()
                    open_list += new_open_list
                    break
                # If there are no more available objects for this transition level, search in the next level
                elif not has_valid_objs and has_next_level:
                    level_counter += 1
                    continue
                # If there are no objects to add to any beam, return the beam with the highest probability
                elif not has_valid_objs and not has_next_level:
                    open_list.sort(key=lambda tup: tup[1], reverse=True)
                    predicted_seq = open_list[0][0]
                    return [ALL_OBJECTS[x - 1] for x in open_list[0][0][1:]]

            if has_valid_objs:
                # Sort and keep only top beam_width beams
                open_list.sort(key=lambda tup: tup[1], reverse=True)
                open_list = open_list[0:beam_width]
        # todo: max seq len not implemented yet
    
    def order_objects_largest(self, unpacked_objs):
        """
        Oreder objects to be packed by size by self.volumes"""

        # throw error if unpacked_objs is None or empty
        if unpacked_objs is None or len(unpacked_objs) == 0:
            raise ValueError('unpacked_objs is None or empty')
        
        # sort the objects by volume
        unpacked_objs = sorted(unpacked_objs, key=lambda x: self.volumes[self.objects.tolist().index(x)], reverse=True)

        return unpacked_objs

    def order_objects(self, unpacked_objs):
        """
        Order objects to be packed. This function is a wrapper for beam_search.
        :param unpacked_objs: Objects that were not yet packed.
        :return: Ordered sequence of objects to be packed.
        """
        # throw error if unpacked_objs is None or empty
        if unpacked_objs is None or len(unpacked_objs) == 0:
            raise ValueError('unpacked_objs is None or empty')
        
        ordered_objs = []
        while len(unpacked_objs)!=0:
    
            next_obj = self.beam_search(unpacked_objs, ALL_TRANSITIONS, 5, 3)
            # remove the next_obj from the unpacked_objs
            unpacked_objs = list((Counter(unpacked_objs) - Counter(next_obj)).elements())
            ordered_objs += next_obj

        return ordered_objs
    
    def simple_reward(self, obj_inside):
        # calculate the reward
        if obj_inside and len(self.unpacked_list) > 0:
            reward = 1
            done = False
        elif obj_inside and len(self.unpacked_list) == 0:
            # reward = 3
            reward = 1
            done = True
        else:
            reward = -1
            done = True
        
        return reward, done

    def compactness_reward(self, obj_inside):
        # calculate the reward
        if obj_inside:
            # get heightmap
            self.space.shot_whole()
            min_box_volume = get_min_box(self.space.heightmapC, self.bin)

            if min_box_volume == 0:
                reward = -1
                done = True
                return reward, done

            compactness = self.cummulative_volume / min_box_volume
            reward = compactness
            
            if len(self.unpacked_list) == 0:
                done = True
            else:
                done = False
        else:
            reward = -1
            done = True
        
        return reward, done
    
    def compactness_stability_reawrd(self, obj_inside):
        # calculate the reward
        if obj_inside:
            # get heightmap
            self.space.shot_whole()
            min_box_volume = get_min_box(self.space.heightmapC, self.bin)
            if min_box_volume == 0:
                reward = -1
                done = True
                return reward, done
            compactness = self.cummulative_volume / min_box_volume

            # check did the object tip over
            # get the rotation of the object
            obj = self.interface.objs[-1]
            rotation = self.interface.get_Wraped_Position_And_Orientation(obj)[1]
            # convert quaternion to Euler
            r = R.from_quat(rotation)
            rotation = r.as_euler('xyz', degrees=True)
            # print('Rotation:', rotation)
            # check if the object tipped over
            if (np.abs(rotation[0]) > 10 and np.abs(rotation[0]) < 170) or (np.abs(rotation[1]) > 10 and np.abs(rotation[1]) < 170):
                stability = 0
            else:
                stability = 1
            # print('Stability:', stability)
            reward = self.alpha * compactness + (1 - self.alpha) * stability

            if len(self.unpacked_list) == 0:
                done = True
            else:
                done = False
        else:
            reward = -1
            done = True
        
        return reward, done
        
    def predict_action_heuristic(self, observation, objname=None, reorient3D=False):
        start_time = time.time()

        if objname is None:
            objname = self.unpacked_list[0]
        
        # spawn the object at -100, -100, -100 and add it to the shapeMap
        # self.interface.addObjectC(name=objname, targetC=[-100, -100, -100], rotation=[0, 0, 0], scale=[1.0, 1.0, 1.0])

        # get the id of the new object
        mesh, _, _ = self.interface.shapeMap[objname]
        meshT = mesh.copy()

        if reorient3D:
            # reorient the mesh
            meshT, rotation = self.reorient_mesh_by_dimensions(meshT)
        else:
            rotation = [0, 0, 0]
        
        # max_length = 0.0

        # for rotation in self.rotation_bins:
        #     meshT = mesh.copy()
        #     # rotate the object
        #     rotation = [0, 0, rotation]
        #     # convert to quat
        #     r = R.from_euler('xyz', rotation, degrees=True)
        #     orientationT = r.as_quat()
        #     meshT.apply_transform(extendMat(transforms3d.euler.quat2mat([orientationT[3], *orientationT[0:3]])))
        #     bounds = meshT.bounds
        #     length = np.abs(bounds[1][0] - bounds[0][0])
        #     if length > max_length:
        #         max_length = length
        #         max_rotation = rotation
        
        # # apply the max_rotation to the object
        # rotation = max_rotation
        # r = R.from_euler('xyz', rotation, degrees=True)
        # orientationT = r.as_quat()
        # meshT.apply_transform(extendMat(transforms3d.euler.quat2mat([orientationT[3], *orientationT[0:3]])))
        bounds = meshT.bounds
        width = int(np.abs(bounds[1][1] - bounds[0][1]) / self.space.resolutionH)
        length = int(np.abs(bounds[1][0] - bounds[0][0]) / self.space.resolutionH)    

        # slide through self.pixel_bins
        min_height = 1000

        i = 0
        for pixel in self.pixel_bins:
            x = pixel[0]
            y = pixel[1]

            if (x+length> 115 or y+width > 76):
                continue

            min_height_t = np.max(observation[0, x:x+length, y:y+width])
            
            if min_height_t < min_height:
                min_height = min_height_t
                best_pixel = pixel
                best_pixel_idx = i
            
            i += 1
        
        # print('Best pixel:', best_pixel)
        # print('Min height:', min_height)
        
        # get the targetC
        

        pose = [best_pixel[0] * self.space.resolutionH, best_pixel[1] * self.space.resolutionH, (min_height/255.0)*0.3 + 0.1]

        targetFLB = pose
        # get euler from orientation quaternion
        # rotation = R.from_quat(orientation).as_euler('xyz', degrees=True)
        time_elapsed = (time.time() - start_time) * 1000
        return targetFLB, rotation, time_elapsed
                

    def get_principal_axes(self, mesh):
        """
        Get the principal axes of a mesh based on its inertia tensor.
        
        Args:
            mesh: A trimesh object
            
        Returns:
            A 3x3 matrix where each row is a principal axis
        """
        # Get the inertia tensor
        inertia = mesh.moment_inertia
        
        # Compute eigenvalues and eigenvectors of the inertia tensor
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        
        # Sort eigenvectors by eigenvalues (ascending)
        # This gives us the principal axes ordered by increasing moment of inertia
        # (which corresponds to the axes ordered by decreasing dimension)
        idx = eigenvalues.argsort()
        # reverse the order
        idx = idx[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Return the principal axes as rows of a matrix
        return eigenvectors.T

    def reorient_mesh_by_dimensions(self, mesh):
        """
        Reorient the mesh so that:
        - smallest dimension is along z-axis (bottom-top)
        - second smallest dimension is along y-axis (left-right)
        - largest dimension is along x-axis (back-front)
        
        Args:
            mesh: A trimesh object
            
        Returns:
            A new trimesh object with the reoriented mesh
        """
        # Copy the mesh to avoid modifying the original
        reoriented_mesh = mesh.copy()
        
        # Get the principal axes
        axes = self.get_principal_axes(mesh)
        
        # Create a rotation matrix to align with desired orientation
        # Note: We want the largest dimension (axis with smallest moment of inertia) along x,
        # second largest along y, and smallest along z
        # The axes are already sorted by descending? moment of inertia (descending dimension)
        rotation_matrix = np.vstack([
            axes[2],  #  largest dimension -> x (back-front) 
            axes[1],  # second largest dimension -> y (left-right)
            axes[0]   # smallest dimension -> z (bottom-top) 
        ])
        

        # convert the rotation matrix to quat
        r = R.from_matrix(rotation_matrix)
        rotation = r.as_quat()
        
        # Apply the rotation
        reoriented_mesh.apply_transform(np.vstack([
            np.hstack([rotation_matrix, np.zeros((3, 1))]),
            [0, 0, 0, 1]
        ]))
        
        return reoriented_mesh, rotation    
    
    def check_stability(self, intended_quat, actual_quat):
        """
        Check if an object has tipped over by comparing intended and actual rotations.
        Rotation around the "up" axis of the intended orientation is allowed.
        
        Parameters:
        intended_quat (np.array): Quaternion [x, y, z, w] representing the intended rotation
        actual_quat (np.array): Quaternion [x, y, z, w] representing the actual rotation
        
        Returns:
        int: 1 if stable (not tipped over), 0 if unstable (tipped over)
        """
        # Create rotation objects from quaternions
        intended_rot = R.from_quat(intended_quat)
        actual_rot = R.from_quat(actual_quat)
        
        # Get the "up" axis in the intended rotation's frame
        # Assuming Z is up in the world frame [0, 0, 1]
        up_vector = np.array([0, 0, 1])
        intended_up = intended_rot.apply(up_vector)
        
        # Get the "up" axis in the actual rotation's frame
        actual_up = actual_rot.apply(up_vector)
        
        # Compute the angle between the two "up" vectors
        dot_product = np.clip(np.dot(intended_up, actual_up), -1.0, 1.0)
        angle_between = np.degrees(np.arccos(dot_product))
        
        # Check if the angle exceeds the threshold (10 degrees)
        # Using the same threshold as in your original code
        if angle_between > 10 and angle_between < 170:
            stability = 0  # Unstable, tipped over
        else:
            stability = 1  # Stable, not tipped over
        
        return stability

if __name__ == '__main__':
        
    with open("test_set_list.json", "rb") as f:
        test_set = json.load(f)
    
    df_results_episode = pd.DataFrame(columns=['filename', 'obj_packed', 'compactness_episode', 'success_episode'])
    df_per_step = pd.DataFrame(columns=['filename', 'compactness_step', 'stability', 'success', 'time_elapsed'])

    env = PackingGameHeuristic(visual=False, ordered_objs='largest', reward_function='compactness', unpacked_list_min=0.7, unpacked_list_max=0.9)

    for filename in test_set:

        unpacked_list = test_set[filename]
        obs, _ = env.reset(unpacked_list=unpacked_list)
        done = False
        obj_packed = 0
        while not done:
            # heightmap = obs[0]
            # draw_heatmap(heightmap)
            targetFLB, rotation, time_elapsed = env.predict_action_heuristic(obs, reorient3D=False)
            obs, done, success, compactness, stability = env.step_with_metrics(targetFLB, rotation)
            
            df_this_step = pd.DataFrame({'filename': filename, 'compactness_step': compactness, 'success': success, 'time_elapsed':time_elapsed}, index=[0])
            df_per_step = pd.concat([df_per_step, df_this_step], ignore_index=True)

        print(f'Filename: {filename}, Objects packed: {obj_packed}, Compactness: {compactness}, Success: {success}')
        df_this_episode = pd.DataFrame({'filename': filename, 'obj_packed': obj_packed, 'compactness_episode': compactness, 'success_episode': success}, index=[0])
        df_results_episode = pd.concat([df_results_episode, df_this_episode], ignore_index=True)
            
    df_results_episode.to_csv('heuristic_stability_test_results_episode_check.csv')
    df_per_step.to_csv('heuristic_heuristic_stability_test_results_step_check.csv')