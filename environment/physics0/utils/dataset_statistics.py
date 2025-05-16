import json
import os
import numpy as np

DASET_FOLDER = 'temp'

success = 0
fail = 0
lenght = []

# for every folder in the dataset folder
for folder in os.listdir(DASET_FOLDER):
    # check is folder a folder
    if not os.path.isdir(os.path.join(DASET_FOLDER, folder)):
        continue
    # load the data
    with open(os.path.join(DASET_FOLDER, folder, folder + 'data.json')) as f:
        data = json.load(f)
    
    done = False
    i = 0

    # for every object in the data
    while not done:
        done = data[i]['done']
        # check if the rotation is violated
        # if data[i]['rotationViolationNew']:
        #     # exit the loop
        #     break
        # check if the object is placed correctly
        if data[i]['done']:
            if data[i]['reward'] > 0:
                success += 1
            else:
                fail += 1
        i += 1
    
    lenght.append(i-1)

print(f'Total count: {success + fail}')
print(f'Success: {success}, Fail: {fail}')
print(f'Mean length: {np.mean(lenght)}')
print(f'Std length: {np.std(lenght)}')
print(f'95% confidence interval:', np.percentile(lenght, 95))
print(f'Percent of success: {success / (success + fail) * 100}%')
        