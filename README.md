# VLN_robot
## Implemented following functions for turtlebot3
* ground projection of occupancy map from depth reading
* semantic segmentation using aihabitat
* semantic segmentation using DeepLabV3
* ground projection of semantic map from semantic output
# pred_ego_crops_sseg[17,:,:] = zero_map

## Completed running the test pipeline of cross-modal map learning for vision and language navigation with real-world data
```
python main.py --name=real --vln_no_map --root_path /home/bo/Desktop/VLN_desktop/aihabitat_data/ --model_exp_dir /home/bo/Desktop/VLN_desktop/VLN_rerun/models/ --save_nav_images --use_first_waypoint
```
## Implemented action-taking commands using ROS for Turtlebot3 and Roomba: move_new_v2.py

## next steps:
1. Set up an indoor house environment (create ground truth semantic map)
2. Perform sim-to-real evaluation of the map, path prediction and VLN
3. Create our own VLN dataset in our real-world setup and retrain the model
