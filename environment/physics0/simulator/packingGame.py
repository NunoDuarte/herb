import gymnasium as gym
import numpy as np
from .Interface import Interface
from environment.physics0.simulator.space import Space, draw_heatmap
from collections import Counter
import pickle
from environment.physics0.utils.volume_utils import get_min_box
from scipy.spatial.transform import Rotation as R
from environment.physics0.simulator.space import draw_heatmap_norm
import time


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

class PackingGame(gym.Env):
    def __init__(self, bin_size=[0.345987, 0.227554, 0.1637639], object_info='dataset/datas/object_info.npz', visual=False, ordered_objs=False, reward_function='simple', alpha=0.9, 
                 unpacked_list_min=0.7, unpacked_list_max=0.9):
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

        if self.ordered_objs:
            self.unpacked_list = self.order_objects(self.unpacked_list)

        self.cummulative_volume = 0.0

        # debugging
        # self.unpacked_list = self.unpacked_list[:3]

        self.space.shot_whole()
        heightmap = self.space.heightmapC
        observation = self.prepare_observation(heightmap, self.projections[self.objects.tolist().index(self.unpacked_list[0])])

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

    def step(self, action, objname=None, actions_normal=False):
        # objname = self.unpacked_list[0]

        if not actions_normal:
            action[0] = unnormalize(action[0], 0.0, 0.345987)
            action[1] = unnormalize(action[1], 0.0, 0.227554)
            action[2] = unnormalize(action[2], 0.0, 180.0)

        # throw error if objname is None
        if objname is None:
            objname = self.unpacked_list[0]
            if objname is None:
                raise ValueError('objname is None')
        
        # take a shot of the scene to find z
        # self.space.shot_whole()

        # rotation = [0, 0, action[2]]
        rotation = [0, 0, action[2]]
        # targetC = [action[0], action[1], 0]
        targetC = [action[0], action[1], 0]
        # PoseT = (targetC, rotation)
        PoseT = (targetC, rotation)
        # print('PoseT:', PoseT)

        mesh, _, _ = self.interface.shapeMap[objname]
        z = self.space.get_object_z(mesh, PoseT)

        targetC[2] = z 
        
        self.interface.addObjectC(name=objname, targetC=targetC, rotation=rotation, scale=[1.0, 1.0, 1.0])
        # simulate the gravity and contact forces
        self.interface.simulateToQuasistatic(linearTol = 0.01, angularTol = 0.01, batch = 1.0, dt = 0.01, maxBatch = 2)  
        # ----------------------------------------    
        self.interface.disableObject(self.interface.objs[-1])

        # remove the placed object
        self.unpacked_list.pop(0)

        # check is the object is inside the bin
        obj_inside = self.interface.check_obj_inside(self.interface.objs[-1], checkZ=0.13)

        self.current_obj_volume = self.volumes[self.objects.tolist().index(objname)]

        if obj_inside:
            self.cummulative_volume += self.current_obj_volume

        self.space.shot_whole()
        self.heightmap_after = self.space.heightmapC

       
        # next objname
        if len(self.unpacked_list) > 0:
            next_objname = self.unpacked_list[0]
        else:
            next_objname = objname

        # prepare the observation
        observation = self.prepare_observation(self.heightmap_after, self.projections[self.objects.tolist().index(next_objname)])

        # calculate the reward
        reward, done = self.reward_function(obj_inside)
        

        return observation, reward, done, False, {}
    
    def step_with_metrics(self, action, objname=None, actions_normal=False):

        start = time.time()
        if not actions_normal:
            action[0] = unnormalize(action[0], 0.0, 0.345987)
            action[1] = unnormalize(action[1], 0.0, 0.227554)
            action[2] = unnormalize(action[2], 0.0, 180.0)

        # throw error if objname is None
        if objname is None:
            objname = self.unpacked_list[0]
            if objname is None:
                raise ValueError('objname is None')
        
        # take a shot of the scene to find z
        # self.space.shot_whole()

        # rotation = [0, 0, action[2]]
        rotation = [0, 0, action[2]]
        # targetC = [action[0], action[1], 0]
        targetC = [action[0], action[1], 0]
        # PoseT = (targetC, rotation)
        PoseT = (targetC, rotation)
        # print('PoseT:', PoseT)

        mesh, _, _ = self.interface.shapeMap[objname]
        z = self.space.get_object_z(mesh, PoseT)

        targetC[2] = z
        
        time_elapsed = (time.time() - start) * 1000

        self.interface.addObjectC(name=objname, targetC=targetC, rotation=rotation, scale=[1.0, 1.0, 1.0])
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
        observation = self.prepare_observation(heightmap, self.projections[self.objects.tolist().index(next_objname)])

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
            
            
            return observation, done, success, compactness, stability, time_elapsed
        
        else:
            success = -1
            done = True
            return observation, done, success, 0, 0, time_elapsed
    
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

    def replay_step_with_metrics(self, targetC, rotation, objname=None):
        # to be used to replay the recorded poses
        # throw error if objname is None
        if objname is None:
            objname = self.unpacked_list[0]
            if objname is None:
                raise ValueError('objname is None')
            
        if len(rotation) == 3:
            # transform the euler to quaternion
            r = R.from_euler('xyz', rotation, degrees=True)
            rotation = r.as_quat()

        self.interface.addObjectC(name=objname, targetC=targetC, rotation=rotation, scale=[1.0, 1.0, 1.0])
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
        observation = self.prepare_observation(heightmap, self.projections[self.objects.tolist().index(next_objname)])

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
            rotation_after = self.interface.get_Wraped_Position_And_Orientation(obj)[1]
            # convert quaternion to Euler
            # r = R.from_quat(rotation_after)
            # rotation_after = r.as_euler('xyz', degrees=True)
            # if (np.abs(rotation_after[0]) > 10 and np.abs(rotation_after[0]) < 170) or (np.abs(rotation_after[1]) > 10 and np.abs(rotation_after[1]) < 170):
            #     stability = 0
            # else:
            #     stability = 1
            stability = self.check_stability(intended_quat=rotation, actual_quat=rotation_after)
            
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

    def render(self, heightname='height.png', colorname='color.png'):
        self.space.shot_whole()
        draw_heatmap(self.space.heightmapC, save=True, savename=heightname, show=False)
        self.interface.saveImage(colorname)

    def close(self):
        self.interface.close()
        self.space.close()

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
            # self.space.shot_whole()
            min_box_volume = get_min_box(self.heightmap_after, self.bin)

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
            # self.space.shot_whole()
            min_box_volume = get_min_box(self.heightmap_after, self.bin)
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
        

if __name__ == '__main__':
    env = PackingGame(visual=True, ordered_objs=True, reward_function='compactness_stability')
    obs, _ = env.reset()
    for i in range(1000):
        # convert obs to image
        heigthmap = obs[0]
        # plot it
        draw_heatmap_norm(heigthmap)
        # wait for input
        # input('Wait')
        obs, reward, done, _, _ = env.step(np.random.uniform(-1, 1, 3))
        print('Reward:', reward)
        

    # print('Mean length:', np.mean(lengths))
    # while True:
    #     env.reset()
    #     done = False
    #     while not done:
    #         _, reward, done, _, _ = env.step(np.random.uniform(-1, 1, 3))
    #         print('Reward:', reward)
    #         # wait for keyboard
    #         input('Wait')
    #         # env.render()
    #         print(env.unpacked_list)
    #         print(len(env.unpacked_list))
    
        
