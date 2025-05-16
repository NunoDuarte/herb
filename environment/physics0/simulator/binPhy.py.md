## Packing Game Environment Documentation 

### Table of Contents 
*  [PackingGame Environment](#packinggame-environment)
    * [Class Initialization](#class-initialization)
    * [Methods](#methods)
        * [non_blocking_simulation()](#non_blocking_simulation)
        * [seed()](#seed)
        * [close()](#close)
        * [reset()](#reset)
        * [get_ratio()](#get_ratio)
        * [get_item_ratio()](#get_item_ratio)
        * [gen_next_item_ID()](#gen_next_item_id)
        * [get_action_candidates()](#get_action_candidates)
        * [get_all_possible_observation()](#get_all_possible_observation)
        * [cur_observation()](#cur_observation)
        * [action_to_position()](#action_to_position)
        * [prejudge()](#prejudge)
        * [step()](#step)
        
### PackingGame Environment 

This Python code defines a `PackingGame` environment class that implements a gym environment for a packing game. The environment allows an agent to pack objects into a bin, maximizing the packing density. 

#### Class Initialization

The `PackingGame` class is initialized with a set of arguments:

| Argument | Description |
|---|---|
| `resolutionA` | Resolution of the action space (in meters). |
| `resolutionH` | Resolution of the height map (in meters). |
| `bin_dimension` | Dimensions of the bin (in meters). |
| `scale` | Scale factor for the environment. |
| `objPath` | Path to the directory containing object meshes. |
| `meshScale` | Scale factor for object meshes. |
| `shapeDict` | Dictionary containing object shape information. |
| `infoDict` | Dictionary containing object information (e.g., volume). |
| `dicPath` | Path to the dictionary file containing object information. |
| `ZRotNum` | Number of rotations along the Z-axis. |
| `heightMap` | Flag indicating whether to use a height map. |
| `only_simulate_current` | Flag indicating whether to only simulate the current object. |
| `selectedAction` | Flag indicating whether to use a pre-selected set of actions. |
| `bufferSize` | Size of the item buffer. |
| `simulation` | Flag indicating whether to perform physics simulation. |
| `evaluate` | Flag indicating whether to evaluate the environment. |
| `maxBatch` | Maximum number of objects to be added to the simulation. |
| `resolutionZ` | Resolution of the Z-axis (for height map). |
| `dataSample` | Type of data sample to use (e.g., 'category', 'instance'). |
| `test_name` | Name of the test data file. |
| `visual` | Flag indicating whether to visualize the environment. |
| `non_blocking` | Flag indicating whether to use non-blocking simulation. |
| `time_limit` | Time limit for non-blocking simulation (in seconds). |


#### Methods

##### `non_blocking_simulation()`

This function is called as a separate thread for non-blocking simulation. It simulates the environment to a quasistatic state.

##### `seed()`

This method sets the random seed for the environment.

##### `close()`

This method closes the environment interface.

##### `reset()`

This method resets the environment to its initial state. It also initializes the `Interface` object, which handles the physics simulation. 

##### `get_ratio()`

This method calculates the packing ratio of the current state.

##### `get_item_ratio()`

This method calculates the ratio of the volume of a specific item to the volume of the bin.

##### `gen_next_item_ID()`

This method generates the ID of the next item to be placed.

##### `get_action_candidates()`

This method gets the possible action candidates for a given item.

##### `get_all_possible_observation()`

This method gets all possible observations for all items in the item buffer.

##### `cur_observation()`

This method returns the current observation of the environment. It includes information about the next item to be placed, the possible positions, and the height map (if enabled).

##### `action_to_position()`

This method converts an action index to the corresponding position and rotation index.

##### `prejudge()`

This method checks if a given position and rotation are valid.

##### `step()`

This method performs a step in the environment. It takes an action as input, simulates the environment, and returns the next observation, reward, done flag, and info dictionary. 

The `step` method implements the following logic: 
1. Convert the action index to the corresponding position and rotation index.
2. Simulate the placement of the object at the specified position and rotation.
3. Update the state of the environment.
4. Return the next observation, reward, done flag, and info dictionary. 

The `step` method also handles non-blocking simulation, where the simulation runs in a separate thread. 
