# Item Creator Documentation

## Table of Contents

| Section | Description |
|---|---|
| [Item Creator Class](#item-creator-class) | Base class for item creators. |
| [Random Item Creator Class](#random-item-creator-class) | Creates random items from a set. |
| [Random Instance Creator Class](#random-instance-creator-class) | Creates random items based on a dictionary of instances. |
| [Random Cate Creator Class](#random-cate-creator-class) | Creates random items based on categories. |
| [Load Item Creator Class](#load-item-creator-class) | Loads items from a pre-defined dataset. |

## Item Creator Class

The `ItemCreator` class is the base class for all item creators. It provides a framework for generating and managing items, which can be used in various applications like data processing or training machine learning models. 

### Class Attributes

* `item_dict`: A dictionary storing the basic code for the item.
* `item_list`: A list storing the generated items.

### Class Methods

* `__init__`: Initializes the `ItemCreator` object.
* `reset`: Clears the `item_list` and resets the internal state.
* `generate_item`: Generates a new item and adds it to the `item_list`. This method must be implemented by subclasses.
* `preview`: Returns a copy of the first `length` items in the `item_list`. If the `item_list` is shorter than `length`, it generates new items until it reaches the desired length.
* `update_item_queue`: Removes the item at the specified index from the `item_list`.


## Random Item Creator Class

The `RandomItemCreator` class is a subclass of `ItemCreator` that generates random items from a given set.

### Class Attributes

* `item_set`: A set of possible items.

### Class Methods

* `__init__`: Initializes the `RandomItemCreator` object.
* `generate_item`: Generates a random item from the `item_set` and adds it to the `item_list`.

## Random Instance Creator Class

The `RandomInstanceCreator` class is a subclass of `ItemCreator` that generates random items based on a dictionary of instances. It selects a random instance from the dictionary and then chooses a random item from the list of items associated with that instance. 

### Class Attributes

* `inverseDict`: A dictionary that maps instance names to lists of corresponding items.

### Class Methods

* `__init__`: Initializes the `RandomInstanceCreator` object.
* `generate_item`: Generates a random item based on the `inverseDict` and adds it to the `item_list`.

## Random Cate Creator Class

The `RandomCateCreator` class is a subclass of `ItemCreator` that generates random items based on categories. It selects a random category with a specified probability and then chooses a random item from the list of items associated with that category.

### Class Attributes

* `categories`: A dictionary storing the probability distribution of each category.
* `objCates`: A dictionary mapping categories to lists of corresponding items.

### Class Methods

* `__init__`: Initializes the `RandomCateCreator` object.
* `generate_item`: Generates a random item based on the `categories` and `objCates` dictionaries and adds it to the `item_list`.

## Load Item Creator Class

The `LoadItemCreator` class is a subclass of `ItemCreator` that loads items from a pre-defined dataset. It iterates through the dataset and generates items sequentially.

### Class Attributes

* `data_name`: The path to the dataset file.
* `traj_index`: The current trajectory index.
* `item_index`: The current item index within the trajectory.
* `item_trajs`: A list of trajectories loaded from the dataset.
* `traj_nums`: The number of trajectories in the dataset.

### Class Methods

* `__init__`: Initializes the `LoadItemCreator` object.
* `reset`: Resets the `item_list` and sets the `traj_index` to a specified value or increments it by one.
* `generate_item`: Generates the next item from the current trajectory and adds it to the `item_list`. If the end of the trajectory is reached, it generates a `None` item. 
