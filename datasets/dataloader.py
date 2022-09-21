
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random
# import habitat
# from habitat.config.default import get_config
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import os
import gzip
import json
from models.semantic_grid import SemanticGrid
import test_utils as tutils
import time
import quaternion
from transformers import BertTokenizer #, BertModel
import math
from models.img_segmentation import get_img_segmentor_from_options
import torch.nn as nn

### Dataloader for storing data in the unknown map case

class HabitatDataVLN_UnknownMap(Dataset):

    # Loads necessary data for the actual VLN task

    def __init__(self, options, existing_episode_list=[], random_poses=False, pose_noise=1):

        self.options = options
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_poses_per_example = options.num_poses_per_example

        self.fx, self.fy = options.fx, options.fy
        self.u0, self.v0 = options.u0, options.v0

        # self.parse_episodes(self.options.datasets)
        
        # self.number_of_episodes = len(self.scene_data["episodes"])

        #print(self.number_of_episodes)
        #print(self.scene_data['episodes'][0])

        # cfg = habitat.get_config(config_file)
        # cfg.defrost()
        # #cfg.SIMULATOR.SCENE = '/media/ggeorgak/DATA/habitat_backup/data/scene_datasets/mp3d/' + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        # cfg.SIMULATOR.SCENE = options.root_path + options.scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        # #cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        # cfg.SIMULATOR.TURN_ANGLE = options.turn_angle
        # cfg.SIMULATOR.FORWARD_STEP_SIZE = options.forward_step_size
        # cfg.freeze()

        # self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        # seed = 0
        # self.sim.seed(seed)

        # self.hfov = float(cfg.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
        # self.cfg_norm_depth = cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
        self.object_labels = options.n_object_classes
        self.spatial_labels = options.n_spatial_classes
        self.global_dim = (options.global_dim, options.global_dim)
        self.grid_dim = (options.grid_dim, options.grid_dim)
        self.cell_size = options.cell_size
        self.heatmap_size = (options.heatmap_size, options.heatmap_size)
        self.num_waypoints = options.num_waypoints
        self.min_angle_noise = np.radians(-15)
        self.max_angle_noise = np.radians(15)
        self.img_size = (options.img_size[0], options.img_size[1])
        self.img_segm_size = (options.img_segm_size[0], options.img_segm_size[1])
        self.normalize = True
        self.pixFormat = 'NCHW'
        #self.crop_size = (options.crop_size, options.crop_size)
        # self.max_depth = cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        # self.min_depth = cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        self.max_depth = 10
        self.min_depth = 0

        #self.preprocessed_scenes_dir = "/media/ggeorgak/DATA/habitat_backup/data/scene_datasets/mp3d_scene_pclouds/"
        self.preprocessed_scenes_dir = options.root_path + options.scenes_dir + "mp3d_scene_pclouds/"

        ### get point cloud and labels of scene
        # self.pcloud, self.label_seq_spatial, self.label_seq_objects = utils.load_scene_pcloud(self.preprocessed_scenes_dir,
        #                                                                                             self.scene_id, self.object_labels)
        # self.color_pcloud = utils.load_scene_color(self.preprocessed_scenes_dir, self.scene_id)

        # Initialize the semantic grid only to use the spatialTransformer. The crop_size (heatmap_size) argument does not matter here
        self.sg = SemanticGrid(1, self.grid_dim, options.heatmap_size, self.cell_size,
                                    spatial_labels=options.n_spatial_classes, object_labels=options.n_object_classes)

        if len(existing_episode_list)!=0:
            self.existing_episode_list = [ int(x.split('_')[2]) for x in existing_episode_list ]
        else:
            self.existing_episode_list=[]
        
        self.random_poses = random_poses
        self.pose_noise = pose_noise # used during store_vln episodes

        self.occupancy_height_thresh = options.occupancy_height_thresh

        # Build 3D transformation matrices for the occupancy egocentric grid
        self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_size[0]), np.linspace(1,-1,self.img_size[1])), device='cuda')
        self.xs = self.xs.reshape(1,self.img_size[0],self.img_size[1])
        self.ys = self.ys.reshape(1,self.img_size[0],self.img_size[1])
        # K = np.array([
        #     [1 / np.tan(self.hfov / 2.), 0., 0., 0.],
        #     [0., 1 / np.tan(self.hfov / 2.), 0., 0.],
        #     [0., 0.,  1, 0],
        #     [0., 0., 0, 1]])
        K = np.array([
            [self.fx, 0., self.u0, 0.],
            [0., self.fy, self.v0, 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])
        print(K)
        # K = np.array([
        #     [self.fx, 0., self.img_size[0]/2, 0.],
        #     [0., self.fy, self.img_size[0]-self.img_size[0]/2, 0.],
        #     [0., 0., 1., 0.],
        #     [0., 0., 0., 1.]])            
        self.inv_K = torch.tensor(np.linalg.inv(K), device='cuda')
        # create the points2D containing all image coordinates
        x, y = torch.tensor(np.meshgrid(np.linspace(0, self.img_size[0]-1, self.img_size[0]), np.linspace(0, self.img_size[1]-1, self.img_size[1])), device='cuda')
        xy_img = torch.vstack((x.reshape(1,self.img_size[0],self.img_size[1]), y.reshape(1,self.img_size[0],self.img_size[1])))
        points2D_step = xy_img.reshape(2, -1)
        self.points2D_step = torch.transpose(points2D_step, 0, 1) # Npoints x 2


        ## Load the pre-trained img segmentation model
        self.img_segmentor = get_img_segmentor_from_options(options)
        self.img_segmentor = self.img_segmentor.to(self.device)
        self.img_segmentor = nn.DataParallel(self.img_segmentor)
        latest_checkpoint = tutils.get_latest_model(save_dir=options.img_segm_model_dir)
        print("Loading image segmentation checkpoint", latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint)
        self.img_segmentor.load_state_dict(checkpoint['models']['img_segm_model'])         
        self.img_segmentor.eval()
        

        ## Build necessary info for ground-projecting the semantic segmentation
        self._xs, self._ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_segm_size[0]), np.linspace(1,-1,self.img_segm_size[1])), device=self.device)
        self._xs = self._xs.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        self._ys = self._ys.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        _x, _y = torch.tensor(np.meshgrid(np.linspace(0, self.img_segm_size[0]-1, self.img_segm_size[0]), 
                                                    np.linspace(0, self.img_segm_size[1]-1, self.img_segm_size[1])), device=self.device)
        _xy_img = torch.cat((_x.reshape(1,self.img_segm_size[0],self.img_segm_size[1]), _y.reshape(1,self.img_segm_size[0],self.img_segm_size[1])), dim=0)
        _points2D_step = _xy_img.reshape(2, -1)
        self._points2D_step = torch.transpose(_points2D_step, 0, 1) # Npoints x 2        


    # def parse_episodes(self, sets):

    #     self.scene_data = {'episodes': []}

    #     for s in sets:

    #         if s=='R2R_VLNCE_v1-2':
    #             root_rxr_dir = self.options.root_path + "rxr-data/" + s + "/"
    #             episode_file = root_rxr_dir + self.options.split + "/" + self.options.split + ".json.gz"
    #             with gzip.open(episode_file, "rt") as fp:
    #                 self.data = json.load(fp)

    #             '''
    #             # count the number of episodes per scene
    #             scene_count={}
    #             dist_list = []
    #             for i in range(len(self.data['episodes'])):
    #                 sc_path = self.data['episodes'][i]['scene_id']
    #                 sc_id = sc_path.split('/')[-1].split('.')[0]
    #                 #start_pos = self.data['episodes'][i]['start_position']
    #                 #goal_pos = self.data['episodes'][i]['goals'][0]['position']
    #                 #if np.absolute(start_pos[1] - goal_pos[1]) >= 0.2:
    #                 #    continue
    #                 #ltg_dist = torch.linalg.norm(torch.tensor(start_pos)-torch.tensor(goal_pos))
    #                 #dist_list.append(ltg_dist.item())
    #                 if sc_id not in scene_count:
    #                     scene_count[sc_id] = 1
    #                 else:
    #                     scene_count[sc_id] += 1
    #             #print(scene_count)
    #             count_all=0
    #             sc_list = list(scene_count.keys())
    #             sc_list.sort()
    #             for sc in sc_list:
    #                 print(sc, scene_count[sc])
    #                 count_all+=scene_count[sc]
    #             #for sc in scene_count.keys():
    #             #    print(sc, scene_count[sc])
    #             #    count_all+=scene_count[sc]
    #             print(len(sc_list))
    #             print("All:", count_all)
    #             #print(dist_list)
    #             #print(np.mean(np.asarray(dist_list)))
    #             #with open('val_seen_dist.npy', 'wb') as f:
    #             #    np.save(f, dist_list)
    #             '''

    #             if self.options.split!="test":
    #                 # Load the gt information from R2R_VLNCE
    #                 episode_file_gt = self.options.root_path+"rxr-data/"+s+"_preprocessed/"+self.options.split +"/"+self.options.split+"_gt.json.gz"
    #                 with gzip.open(episode_file_gt, "rt") as fp:
    #                     self.data_gt = json.load(fp)
                
    #             #print(len(self.data['episodes']))
    #             #print(len(self.data_gt['episodes']))
    #             #print(len(self.data_gt.keys()))
    #             #print(self.data['episodes'][300])
    #             #print()
    #             #print(self.data_gt['episodes'][300])
    #             #self.instruction_vocab = self.data['instruction_vocab']
    #             # Need to keep only episodes that belong to current scene
    #             for i in range(len(self.data['episodes'])):
    #                 sc_path = self.data['episodes'][i]['scene_id']
    #                 sc_id = sc_path.split('/')[-1].split('.')[0]
    #                 if sc_id == self.scene_id:                        
    #                     # Check if given path has enough poses
    #                     #ref_path = self.data['episodes'][i]['reference_path']
    #                     #if len(ref_path) < self.num_poses_per_example:
    #                     #    continue
                        
    #                     # seems that the "start_rotation" for the R2R_VLNCE_v1-2 set has 180 degrees difference from the description
    #                     #self.data['episodes'][i]['start_rotation'] = utils.add_to_quarternion(rotation=self.data['episodes'][i]['start_rotation'], angle=-np.pi)
    #                     self.data['episodes'][i]['scene_id'] = self.scene_id
    #                     self.data['episodes'][i]['dataset'] = s
                        
    #                     if self.options.split!="test":
    #                         # check whether goal is at the same height as start position
    #                         start_pos = self.data['episodes'][i]['start_position']
    #                         goal_pos = self.data['episodes'][i]['goals'][0]['position']
    #                         if np.absolute(start_pos[1] - goal_pos[1]) >= 0.2 and self.options.check_floor:
    #                             continue
    #                         # get gt info
    #                         gt_info = self.data_gt[ str(self.data['episodes'][i]['episode_id']) ] # locations, forward_steps, actions
    #                         #print(len(gt_info['locations']), gt_info['forward_steps'], len(gt_info['actions']))
    #                         self.data['episodes'][i]['waypoints'] = gt_info['locations']
    #                         self.data['episodes'][i]['actions'] = gt_info['actions']

    #                     self.scene_data['episodes'].append(self.data['episodes'][i])
                            

    def __len__(self):
        return self.number_of_episodes


    def get_covered_waypoints(self, waypoints_pose_coords, pose_coords):
        covered = torch.zeros((len(waypoints_pose_coords)))
        dist = np.linalg.norm(waypoints_pose_coords.cpu().numpy() - pose_coords.cpu().numpy(), axis=-1)
        ind = np.argmin(dist)
        covered[:ind] = 1
        return covered


    def sample_random_poses(self, episode):
        idx_pos = random.sample(list(range(len(episode['waypoints']))), self.num_poses_per_example)
        idx_pos.sort()
        idx_pos[0] = 0 # always include the initial position
        init_positions = np.asarray(episode['waypoints'])[idx_pos]
        sim_positions = np.zeros((init_positions.shape[0],3))
        # add noise to the positions, need to check whether the new location is navigable
        for i in range(len(init_positions)):
            valid=False
            while not valid:
                x_noise = np.random.uniform(low=-self.pose_noise, high=self.pose_noise, size=1)
                z_noise = np.random.uniform(low=-self.pose_noise, high=self.pose_noise, size=1)
                loc = init_positions[i].copy()
                loc[0] = loc[0] + x_noise
                loc[2] = loc[2] + z_noise
                if self.sim.is_navigable(loc):
                    valid=True
            sim_positions[i,:] = loc
        # randomly select the orientations
        theta_rand = np.random.uniform(low=-np.pi, high=np.pi, size=len(sim_positions))
        sim_rotations = []
        for k in range(len(sim_positions)):
            sim_rotations.append( quaternion.from_euler_angles([0, theta_rand[k], 0]) )
        sim_positions = sim_positions.tolist()
        return sim_positions, sim_rotations


    # For the unknown map case we need to store the following:
    # instruction, waypoint heatmaps, visible, covered
    # RGB, Depth, semantic observations (use the pretrained img segmentor here directly to get the ground-projected semantics)
    # egocentric incomplete occupancy (input)
    # egocentric gt occupancy
    # egocentric gt semantics


    def __getitem__(self, idx):
        #idx = 11 # ** tmp hardcode
        #print("Episode", idx)
        
        #if idx in self.existing_episode_list:
        #    print("Episode", idx, 'already exists!')
        #    return None
        
        episode = self.scene_data['episodes'][idx]
        #print(episode)

        # check that the episode is not too large to fit in memory
        #if len(episode['actions']) > 50:
        #    return None

        instruction = episode['instruction']['instruction_text']
        #print(instruction)
        
        init_waypoints = episode['waypoints']#[1:] # first waypoint is start position
        actions = episode['actions'][:-1] # remove last action (stop)
        #print("Actions:", len(actions))
        #print(episode['reference_path'])
        goal_position = episode['goals'][0]['position']
        #print('Start:', episode["start_position"], "Goal:", goal_position)
        #print('Rotation:', episode["start_rotation"])

        k = math.ceil(len(init_waypoints) / (self.num_waypoints))
        waypoints = init_waypoints[::k]

        if len(waypoints) == self.num_waypoints:
            waypoints = waypoints[:-1]
            waypoints.append(goal_position) # remove last point and put the goal
        else:
            while len(waypoints) < self.num_waypoints:
                waypoints.append(goal_position)
        #print(waypoints)
        #print(len(waypoints))

        if len(waypoints) > self.num_waypoints:
            raise Exception('Waypoints contains more than '+str(self.num_waypoints))


        # Get the 3D scene semantics
        scene = self.sim.semantic_annotations()
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        # convert the labels to the reduced set of categories
        instance_id_to_label_id_occ = instance_id_to_label_id.copy()
        instance_id_to_label_id_sem = instance_id_to_label_id.copy()
        for inst_id in instance_id_to_label_id.keys():
            curr_lbl = instance_id_to_label_id[inst_id]
            instance_id_to_label_id_occ[inst_id] = viz_utils.label_conversion_40_3[curr_lbl]
            instance_id_to_label_id_sem[inst_id] = viz_utils.label_conversion_40_27[curr_lbl]      


        # set simulator pose at episode start
        self.sim.reset()
        self.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
        sim_obs = self.sim.get_sensor_observations()
        observations = self.sim._sensor_suite.get_observations(sim_obs)

        # To sample locations with noise, randomly select 10 locations from episode['waypoints']
        # and randomly select orientation and add noise to the position. Move the simulator directly to those locations
        if self.random_poses:
            sim_positions, sim_rotations = self.sample_random_poses(episode)
            iterations = len(sim_positions)
        else:
            iterations = len(actions)

        abs_poses = []
        rel_abs_poses = []
        local3D = []
        #goal_rel_pose_coords = torch.zeros((len(actions), 2), dtype=torch.float32, device=self.device)
        #ego_occ_maps = torch.zeros((iterations, self.spatial_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        gt_maps = torch.zeros((iterations, 1, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        gt_maps_occ = torch.zeros((iterations, 1, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        ego_segm_maps = torch.zeros((iterations, self.object_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        goal_maps = torch.zeros((iterations, self.num_waypoints, self.heatmap_size[0], self.heatmap_size[1]), dtype=torch.float32, device=self.device)
        visible_waypoints = torch.zeros((iterations, self.num_waypoints))
        covered_waypoints = torch.zeros((iterations, self.num_waypoints))

        ### Get egocentric map at each waypoint location along with its corresponding relative goal
        for t in range(iterations):

            if self.random_poses:
                self.sim.set_agent_state(sim_positions[t], sim_rotations[t])
                sim_obs = self.sim.get_sensor_observations()
                observations = self.sim._sensor_suite.get_observations(sim_obs)

            img = observations['rgb'][:,:,:3]
            depth_obsv = observations['depth'].permute(2,0,1).unsqueeze(0)

            depth = F.interpolate(depth_obsv.clone(), size=self.img_size, mode='nearest')
            depth = depth.squeeze(0).permute(1,2,0) 

            # if self.cfg_norm_depth:
            #     depth = utils.unnormalize_depth(depth, min=self.min_depth, max=self.max_depth)

            local3D_step = utils.depth_to_3D(depth, self.img_size, self.xs, self.ys, self.inv_K)
            local3D.append(local3D_step)

            agent_pose, y_height = utils.get_sim_location(agent_state=self.sim.get_agent_state())
            abs_poses.append(agent_pose)

            # get the relative pose with respect to the first pose in the sequence
            rel_abs_pose = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
            rel_abs_poses.append(rel_abs_pose)

            ### Run the img segmentation model to get the ground-projected semantic segmentation
            depth_img = depth.permute(2,0,1).unsqueeze(0)
            depth_img = F.interpolate(depth_img, size=self.img_segm_size, mode='nearest')
            
            imgData = utils.preprocess_img(img, cropSize=self.img_segm_size, pixFormat=self.pixFormat, normalize=self.normalize)
            segm_batch = {'images':imgData.to(self.device).unsqueeze(0).unsqueeze(0),
                        'depth_imgs':depth_img.to(self.device).unsqueeze(0)}

            pred_ego_crops_sseg, _ = utils.run_img_segm(model=self.img_segmentor, 
                                                    input_batch=segm_batch, 
                                                    object_labels=self.object_labels, 
                                                    crop_size=self.grid_dim, 
                                                    cell_size=self.cell_size,
                                                    xs=self._xs,
                                                    ys=self._ys,
                                                    inv_K=self.inv_K,
                                                    points2D_step=self._points2D_step)            
            pred_ego_crops_sseg = pred_ego_crops_sseg.squeeze(0)
            ego_segm_maps[t,:,:,:] = pred_ego_crops_sseg
            #viz_utils.write_tensor_imgSegm(img=pred_img_segm, savepath='', name='pred_img_segm')
            #viz_utils.write_tensor_imgSegm(img=pred_ego_crops_sseg.cpu(), savepath='', name='pred_ego_crops'+str(t))
            #viz_utils.write_img(img=imgData.unsqueeze(0), savepath='', name='rgb'+str(t))


            ### Get gt map from agent pose (pose is at the center looking upwards)
            x, y, z, label_seq, color_pcloud = map_utils.slice_scene(x=self.pcloud[0].copy(),
                                                                y=self.pcloud[1].copy(),
                                                                z=self.pcloud[2].copy(),
                                                                label_seq=self.label_seq_objects.copy(),
                                                                height=y_height,
                                                                color_pcloud=self.color_pcloud)
            gt_map_semantic, _ = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[t],
                                                        grid_dim=self.grid_dim, cell_size=self.cell_size, color_pcloud=color_pcloud, z=z)
            #print(gt_map_semantic.shape)
            #print(gt_map_color.shape)
            gt_maps[t,:,:,:] = gt_map_semantic
            #viz_utils.write_tensor_imgSegm(img=gt_map_semantic.unsqueeze(0).cpu(), savepath="", name="gt_map_semantic"+str(t))
            #viz_utils.write_img(img=gt_map_color.unsqueeze(0).cpu(), savepath="", name="gt_map_color"+str(t))
            
            # get gt map from agent pose (pose is at the center looking upwards)
            x, y, z, label_seq_occ = map_utils.slice_scene(x=self.pcloud[0].copy(),
                                                                y=self.pcloud[1].copy(),
                                                                z=self.pcloud[2].copy(),
                                                                label_seq=self.label_seq_spatial.copy(),
                                                                height=y_height)
            gt_map_occupancy = map_utils.get_gt_map(x, y, label_seq_occ, abs_pose=abs_poses[t],
                                                        grid_dim=self.grid_dim, cell_size=self.cell_size, z=z)
            gt_maps_occ[t,:,:,:] = gt_map_occupancy
            #viz_utils.write_tensor_imgSegm(img=gt_map_occupancy.unsqueeze(0).cpu(), savepath="", name="gt_map_occupancy"+str(t), labels=3)

            # get the relative pose with respect to the first pose in the sequence
            rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[t])
            _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
            _rel_pose = _rel_pose.to(self.device)
            pose_coords = tutils.get_coord_pose(self.sg, _rel_pose, abs_poses[t], self.grid_dim[0], self.cell_size, self.device) # B x T x 3
            #print(pose_coords) # should always be in the middle of the map

            # Transform waypoints with respect to agent current pose
            waypoints_pose_coords = torch.zeros((len(waypoints), 2))
            for k in range(len(waypoints)):
                point_pose_coords, visible = tutils.transform_to_map_coords(sg=self.sg, position=waypoints[k], abs_pose=abs_poses[t], 
                                                                                        grid_size=self.grid_dim[0], cell_size=self.cell_size, device=self.device)
                waypoints_pose_coords[k,:] = point_pose_coords.squeeze(0).squeeze(0)
                visible_waypoints[t,k] = visible

            # Find the waypoints already covered in the episode
            covered_waypoints[t,:] = self.get_covered_waypoints(waypoints_pose_coords, pose_coords.squeeze(0))

            waypoints_heatmaps = utils.locs_to_heatmaps(keypoints=waypoints_pose_coords, img_size=self.grid_dim, out_size=self.heatmap_size)
            #print(waypoints_heatmaps.shape)
            goal_maps[t,:,:,:] = waypoints_heatmaps

            #viz_utils.vis_episode(gt_map_semantic=gt_maps[t,:,:,:], pose_coords=waypoints_pose_coords, name=str(t)+"_"+episode['scene_id'])
            #viz_utils.vis_episode(gt_map_semantic=gt_maps_occ[t,:,:,:], pose_coords=waypoints_pose_coords, name=str(t)+"_"+episode['scene_id'], color_mapping=3)
            #viz_utils.vis_heatmaps(goal_maps[t,9,:,:], goal_maps[t,9,:,:])
            #viz_utils.save_map_goal(gt_maps[t,:,:,:].unsqueeze(0).unsqueeze(0), pose_coords, waypoints_pose_coords[-1].unsqueeze(0).unsqueeze(0), "", t)

            #goal_rel_pose_coords[t,:] = goal_pose_coords.squeeze(1)
            #print(goal_rel_pose_coords)

            if not self.random_poses:
                action_id = actions[t]
                observations = self.sim.step(action_id)


        abs_poses = torch.from_numpy(np.asarray(abs_poses)).float()
        rel_abs_poses = torch.from_numpy(np.asarray(rel_abs_poses)).float()

        ### Get the ground-projected observation from the accumulated projected map
        ego_grid_maps = map_utils.est_occ_from_depth(local3D, grid_dim=self.global_dim, cell_size=self.cell_size, 
                                                    device=self.device, occupancy_height_thresh=self.occupancy_height_thresh)
        step_ego_grid_maps = map_utils.get_acc_proj_grid(ego_grid_maps, rel_abs_poses, abs_poses, self.grid_dim, self.cell_size)
        step_ego_grid_maps = map_utils.crop_grid(grid=step_ego_grid_maps, crop_size=self.grid_dim)
        #viz_utils.write_tensor_image(grid=step_ego_grid_maps, savepath="", name="step_ego_grid_occ", sseg_labels=3)


        if not self.random_poses:
            # Sample snapshots along the episode to get num_poses_per_example size
            k = math.ceil(len(actions) / (self.num_poses_per_example))
            inds = list(range(0,len(actions),k))
            avail = list(set(list(range(0,len(actions)))) - set(inds))
            while len(inds) < self.num_poses_per_example:
                inds.append(random.sample(avail, 1)[0])
            while len(inds) > self.num_poses_per_example:
                inds = inds[-1]
            inds.sort()
            abs_poses = abs_poses[inds, :] # num_poses x 3
            goal_maps = goal_maps[inds, :, :, :] # num_poses x num_waypoints x 64 x 64
            step_ego_grid_maps = step_ego_grid_maps[inds, :, :, :] # num_poses x spatial_labels x 192 x 192
            ego_segm_maps = ego_segm_maps[inds, :, :, :]
            gt_maps = gt_maps[inds, :, :, :] # num_poses x 1 x 192 x 192
            gt_maps_occ = gt_maps_occ[inds, :, :, :]
            visible_waypoints = visible_waypoints[inds, :] # num_poses x num_waypoints
            covered_waypoints = covered_waypoints[inds, :] # num_poses x num_waypoints


        item = {}
        item['goal_heatmap'] = goal_maps #goal_heatmap
        item['step_ego_grid_maps'] = step_ego_grid_maps
        item['ego_segm_maps'] = ego_segm_maps
        item['map_semantic'] = gt_maps #gt_map_semantic.cpu()
        item['map_occupancy'] = gt_maps_occ
        #item['tokens'] = torch.tensor(tokens) # list of word tokens corresponding to vocabulary
        #item['text_feat'] = torch.from_numpy(instruction_feat)
        item['abs_pose'] = abs_poses
        #item['goal_position'] = goal_position # absolute goal position, consistent within an episode
        #item['goal_rel_pose_coords'] = goal_rel_pose_coords # relative map coords for each goal (agent is at [255,255])
        item['instruction'] = instruction #episode['instruction']['instruction_text']
        item['visible_waypoints'] = visible_waypoints
        item['covered_waypoints'] = covered_waypoints

        item['dataset'] = episode['dataset']
        item['episode_id'] = episode['episode_id']
        
        return item