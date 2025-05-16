## PyBullet Simulation Interface Documentation 

This document provides an internal code documentation for the Python class `Interface`, designed to interact with PyBullet for simulating 3D objects within a virtual environment. 

### Table of Contents

- [Introduction](#introduction)
- [Class `Interface`](#class-interface)
    - [Constructor (`__init__`)](#constructor-init)
    - [Methods](#methods)
        - [`close()`](#close)
        - [`removeBody()`](#removebody)
        - [`reset()`](#reset)
        - [`getAllPositionAndOrientation()`](#getallpositionandorientation)
        - [`makeBox()`](#makebox)
        - [`addBox()`](#addbox)
        - [`overlap2d()`](#overlap2d)
        - [`adjustHeight()`](#adjustheight)
        - [`addObject()`](#addobject)
        - [`simulatePlain()`](#simulateplain)
        - [`simulateToQuasistatic()`](#simulatetoquasistatic)
        - [`simulateToQuasistaticRecord()`](#simulatetoquasistaticrecord)
        - [`secondSimulation()`](#secondsimulation)
        - [`simulateHeight()`](#simulateheight)
        - [`disableObject()`](#disableobject)
        - [`enableObjects()`](#enableobjects)
        - [`disableAllObject()`](#disableallobject)
        - [`cameraForRecord()`](#cameraforrecord)
        - [`setupCamera()`](#setupcamera)
        - [`get_wraped_AABB()`](#get_wraped_aabb)
        - [`get_Wraped_Position_And_Orientation()`](#get_wraped_position_and_orientation)
        - [`reset_Wraped_Position_And_Orientation()`](#reset_wraped_position_and_orientation)
        - [`reset_Height()`](#reset_height)
        - [`get_trimesh_AABB()`](#get_trimesh_aabb)
        - [`get_trimesh_Position_And_Orientation()`](#get_trimesh_position_and_orientation)
        - [`reset_trimesh_Position_And_Orientation()`](#reset_trimesh_position_and_orientation)
        - [`reset_trimesh_height()`](#reset_trimesh_height)
        - [`reset_trimesh_Position_And_Orientation_new()`](#reset_trimesh_position_and_orientation_new)
- [Helper Functions](#helper-functions)
    - [`extendMat()`](#extendmat)


### Introduction 

This class provides an interface to interact with PyBullet, a physics engine used for simulating objects in a 3D environment. It allows for loading, adding, manipulating, and simulating objects within a virtual box.

### Class `Interface`

The `Interface` class acts as a central point for all interactions with the PyBullet simulation environment. 

#### Constructor (`__init__`)

The constructor initializes various parameters and settings for the simulation environment.

| Parameter | Description | Type | Default |
|---|---|---|---|
| `bin` | Dimensions of the virtual box (length, width, height) | list | [10, 10, 5] |
| `foldername` | Directory to store simulation data | string | '../dataset/datas/128' |
| `visual` | Enable visualization (GUI) for simulation | bool | False |
| `scale` | Scaling factor applied to objects | list | [1.0, 1.0, 1.0] |
| `simulationScale` | Scaling factor for simulation (different from `scale`) | float | None |
| `maxBatch` | Maximum number of simulation steps per batch | int | 2 |

**Initialization Steps:**

1. **Connect to PyBullet:** Connect to either shared memory, GUI, or direct connection based on `visual` flag.
2. **Set Default Scale:** Store the `scale` value as `defaultScale`.
3. **Set Simulation Scale:** Use provided `simulationScale` or default to 1 if `None`.
4. **Initialize Bin:** Store the bin dimensions as a NumPy array and apply `defaultScale`.
5. **Create Shape Map:** Initialize an empty dictionary to store object shapes (for efficient re-use).
6. **Initialize Object Lists:** Create empty lists to store simulated objects (`objs`) and dynamic objects (`objsDynamic`).
7. **Set Gravity:** Set the simulation gravity to [0, 0, -10].
8. **Configure Physics Engine Parameters:** Adjust PyBullet parameters for constraint solver, global CFM, and solver iterations.
9. **Create Virtual Box:** Add a box with the defined dimensions (`bin`) to the simulation environment.
10. **Enable Visualization:** If `visual` is True, enable rendering and disable GUI and tiny renderer.
11. **Create Container Folder:** Create a folder to store simulation data based on the bin dimensions.
12. **Initialize AABB Compensation:** Define a small value for compensating in AABB calculations.
13. **Set up Camera:** Configure the camera for recording the simulation.
14. **Initialize Mesh Dictionary:** Create an empty dictionary to store object meshes.

#### Methods

The class `Interface` provides a comprehensive set of methods for managing and interacting with the PyBullet simulation environment.

**1. `close()`**
- Disconnects from the PyBullet server.

**2. `removeBody()`**
- Removes a specific body (object) from the simulation by its ID.

**3. `reset()`**
- Removes all objects from the simulation environment, resetting the state.

**4. `getAllPositionAndOrientation()`**
- Returns a list of positions and orientations of all simulated objects.

**5. `makeBox()`**
- Creates a box mesh with specified dimensions, color, and thickness.
- Returns a list of box sides as `trimesh` objects. 

**6. `addBox()`**
- Adds a box to the simulation environment with defined dimensions, scaling factor, and position shift. 

**7. `overlap2d()`**
- Checks for 2D overlap between two rectangles defined by their minimum and maximum corner points.

**8. `adjustHeight()`**
- Adjusts the height of a specific object in the simulation.

**9. `addObject()`**
- Adds a new object to the simulation environment.
- The object can be specified by its name, target position, rotation, scale, density, damping values, file path, and color.
- Returns the ID of the newly created object.

**10. `simulatePlain()`**
- Simulates the environment for a specified number of steps with a fixed timestep.

**11. `simulateToQuasistatic()`**
- Simulates the environment until a quasi-static state is reached, where objects are at rest. 
- Optionally specifies a target object ID for simulation.
- Returns True if simulation succeeded, False otherwise.

**12. `simulateToQuasistaticRecord()`**
- Simulates the environment to a quasi-static state and records object positions and orientations during simulation. 
- Optionally specifies a target object ID or a list of objects for simulation.

**13. `secondSimulation()`**
- Performs a secondary simulation to further stabilize the environment, enabling objects and simulating for a specified duration.

**14. `simulateHeight()`**
- Simulates the environment to check if a specific object has reached the bottom of the box.

**15. `disableObject()`**
- Disables physics interaction for a specific object, effectively making it static.
- Optionally sets a target height for the object. 

**16. `enableObjects()`**
- Enables physics interaction for all objects in the simulation.

**17. `disableAllObject()`**
- Disables physics interaction for all objects in the simulation.

**18. `cameraForRecord()`**
- Sets up the camera for recording simulations.

**19. `setupCamera()`**
- Configures the camera parameters (distance, yaw, pitch, target position) within the simulation.

**20. `get_wraped_AABB()`**
- Returns the Axis-Aligned Bounding Box (AABB) of a specific object, potentially applying scaling.

**21. `get_Wraped_Position_And_Orientation()`**
- Returns the position and orientation of an object, potentially applying scaling.

**22. `reset_Wraped_Position_And_Orientation()`**
- Resets the position and orientation of an object to specified target values.

**23. `reset_Height()`**
- Resets the height of an object to a specific target value.

**24. `get_trimesh_AABB()`**
- Returns the AABB of an object based on its `trimesh` representation.

**25. `get_trimesh_Position_And_Orientation()`**
- Returns the position and orientation of an object based on its `trimesh` representation.

**26. `reset_trimesh_Position_And_Orientation()`**
- Resets the position and orientation of an object based on its `trimesh` representation.

**27. `reset_trimesh_height()`**
- Resets the height of an object based on its `trimesh` representation.

**28. `reset_trimesh_Position_And_Orientation_new()`**
- Resets the position and orientation of an object based on its `trimesh` representation, using a new approach.

### Helper Functions

**1. `extendMat()`**
- Extends a 3x3 matrix to a 4x4 matrix, optionally adding a translation vector.
- Used to convert rotation matrices for PyBullet.
