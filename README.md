## Implemented following functions for turtlebot3
* ground projection of occupancy map from depth reading
* semantic segmentation using aihabitat
* semantic segmentation using DeepLabV3
* ground projection of semantic map from semantic output
* updated semantic map with occupancy map

## Installation:
Please refer to: https://github.com/ggeorgak11/CM2

## Completed running the test pipeline of cross-modal map learning for vision and language navigation with real-world data
To run the data, first create the ROS working space following the steps described in:
https://github.com/BobWu1998/Vision-Language-on-Roomba

After creating the workspace, git clone this repository to
```path_to_workspace/src/move_cmds```

To make the code recognizable for ROS, rename the folder when git clone it
```git clone "name of this repo" src```

Don't forget to make the main function readable for the user:
```chmod u+r main.py```

Run the following command:
```
python main.py --name=test_episode --vln_no_map --root_path aihabitat_data/ --model_exp_dir models/ --save_nav_images --use_first_waypoint \
--steps_after_plan 1
```
## Implemented action-taking commands using ROS for Turtlebot3 and Roomba: move_new_v2.py

## next steps:
1. Set up an indoor house environment (create ground truth semantic map)
2. Perform sim-to-real evaluation of the map, path prediction and VLN
3. Create our own VLN dataset in our real-world setup and retrain the model
