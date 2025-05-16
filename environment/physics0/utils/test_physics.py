#--coding:utf-8--
from simulator.Interface import Interface
from simulator.space import Space, draw_heatmap
import numpy as np

bin = np.array([0.345987, 0.227554, 0.1637639])
interface = Interface(visual=True, bin=bin)
space = Space(bin, resolutionAct=0.003, resolutionH=0.003, ZRotNum=8, scale=[1.0, 1.0, 1.0], cut_last_row=True)

# ALL OBJECTS LIST
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

projections = []

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


# spawn all objects at -100, -100, -100 to add the to the shapeMap
for obj in ALL_OBJECTS:
    print(obj)
    interface.addObjectC(name=obj, targetC=[-100, -100, -100], rotation=[0, 0, 0], scale=[1.0, 1.0, 1.0])

interface.reset()

# while True:
for obj in ALL_OBJECTS:
    space.shot_whole()
    targetC = [bin[0]/2,  bin[1]/2, 0]
    rotation = [0, 0, 0]
    # interface.addObjectC(name='011 Banana', targetC=targetC, rotation=rotation, scale=[1.0, 1.0, 1.0])
    PoseT = (targetC, rotation)
    mesh, _, _ = interface.shapeMap[obj]
    z = space.get_object_z(mesh, PoseT)
    targetC[2] = z
    interface.addObjectC(name=obj, targetC=targetC, rotation=rotation, scale=[1.0, 1.0, 1.0])
    # interface.disableObject(interface.objs[-1])
    # simulate
    if obj != "057 Racquetball" and obj != "058 Golf Ball":
        interface.simulateToQuasistatic(linearTol = 0.01, angularTol = 0.01, batch = 1.0, dt = 0.01, maxBatch = 2)
    else:
        interface.simulateToQuasistatic(linearTol = 0.01, angularTol = 0.01, batch = 1.0, dt = 0.01, maxBatch = 1)
    # disable the object
    interface.disableObject(interface.objs[-1])
    # check if the object is inside the bin
    print(interface.check_obj_inside(interface.objs[-1], checkZ=0.1))
    # destroy the object
    # space.reset()
    # interface.reset()
    # take a shot
    space.shot_whole()
    # min_box_volme = get_compactness(space.heightmapC, bin)
    # print(min_box_volme)
    # print(space.heightmapC.shape)
    #print max of heightmapC
    # print(np.max(space.heightmapC))
    # print min of heightmapC
    # print(np.min(space.heightmapC))
    heightmap = space.heightmapC[20:92, 2:74]
    projections.append(heightmap)
    draw_heatmap(heightmap)
    # print(interface.shapeMap)
    # print(interface.objs)
    interface.reset()
    interface.simulatePlain()

# save a npz file of objects and their projections
# np.savez('objects_projections.npz', objects=ALL_OBJECTS, projections=projections)

# take an image of the scene
# interface.saveImage('test.png')

space.shot_whole()
draw_heatmap(space.heightmapC)



while True:
    interface.addObject(name='006 Mustard Bottle_3')
    interface.simulateToQuasistatic()
interface.close()  



        