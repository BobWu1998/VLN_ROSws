
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random
import habitat
from habitat.config.default import get_config
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


class HabitatDataOffline(Dataset):

    def __init__(self, options, config_file, img_segm=False, finetune=False):
        config = get_config(config_file)
        self.config = config
        
        self.img_segm = img_segm
        self.finetune = finetune # whether we are running a finetuning active job

        self.episodes_file_list = []
        self.episodes_file_list += self.collect_stored_episodes(options, split=config.DATASET.SPLIT)
        
        if options.dataset_percentage < 1: # Randomly choose the subset of the dataset to be used
            random.shuffle(self.episodes_file_list)
            self.episodes_file_list = self.episodes_file_list[ :int(len(self.episodes_file_list)*options.dataset_percentage) ]
        self.number_of_episodes = len(self.episodes_file_list)

        self.object_labels = options.n_object_classes

        if self.img_segm:
            self.episodes_imgSegm_dir = options.stored_imgSegm_episodes_dir
            self.episodes_dir = options.stored_episodes_dir


    def collect_stored_episodes(self, options, split):
        episodes_dir = options.stored_episodes_dir + split + "/"
        episodes_file_list = []
        _scenes_dir = os.listdir(episodes_dir)
        scenes_dir = [ x for x in _scenes_dir if os.path.isdir(episodes_dir+x) ]
        for scene in scenes_dir:
            for fil in os.listdir(episodes_dir+scene+"/"):
                episodes_file_list.append(episodes_dir+scene+"/"+fil)
        return episodes_file_list


    def __len__(self):
        return self.number_of_episodes


    def __getitem__(self, idx):
        # Load from the pre-stored objnav training episodes
        ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        abs_pose = ep['abs_pose']
        ego_grid_crops_spatial = torch.from_numpy(ep['ego_grid_crops_spatial'])
        step_ego_grid_crops_spatial = torch.from_numpy(ep['step_ego_grid_crops_spatial'])
        gt_grid_crops_spatial = torch.from_numpy(ep['gt_grid_crops_spatial'])
        gt_grid_crops_objects = torch.from_numpy(ep['gt_grid_crops_objects'])

        ### Transform abs_pose to rel_pose
        rel_pose = []
        for i in range(abs_pose.shape[0]):
            rel_pose.append(utils.get_rel_pose(pos2=abs_pose[i,:], pos1=abs_pose[0,:]))

        item = {}
        item['pose'] = torch.from_numpy(np.asarray(rel_pose)).float()
        item['abs_pose'] = torch.from_numpy(abs_pose).float()
        item['ego_grid_crops_spatial'] = ego_grid_crops_spatial # already torch.float32
        item['step_ego_grid_crops_spatial'] = step_ego_grid_crops_spatial
        item['gt_grid_crops_spatial'] = gt_grid_crops_spatial # Long tensor, int64
        item['gt_grid_crops_objects'] = gt_grid_crops_objects # Long tensor, int64


        if self.img_segm:

            if self.finetune:
                item['images'] = torch.from_numpy(ep['images']) # T x 3 x H x W # images are already pre-processed
                item['gt_segm'] = torch.from_numpy(ep['ssegs']).type(torch.int64) # T x 1 x H x W
                item['depth_imgs'] = torch.from_numpy(ep['depth_imgs']) # T x 1 x H x W
            else:
                ep_file_imgSegm = ep_file.replace(self.episodes_dir, self.episodes_imgSegm_dir)
                ep_imgSegm = np.load(ep_file_imgSegm)
                pred_ego_crops_sseg = torch.from_numpy(ep_imgSegm['pred_ego_crops_sseg'])
                item['pred_ego_crops_sseg'] = pred_ego_crops_sseg

        return item


# Dataloader only for training the img segmentation (i.e. loading only relevant data) that inherits from HabitatDataOffline
class HabitatDataImgSegm(HabitatDataOffline):

    def __init__(self, options, config_file, store=False):
        super().__init__(options, config_file, img_segm=False)
        self.store = store
        self.hfov = float(self.config.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180. # used during store_img_segm_ep.py

    def __getitem__(self, idx):
        # Load from the pre-stored objnav training episodes
        ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        item={}
        item['images'] = torch.from_numpy(ep['images']) # T x 3 x H x W # images are already pre-processed
        item['gt_segm'] = torch.from_numpy(ep['ssegs']).type(torch.int64) # T x 1 x H x W
        item['depth_imgs'] = torch.from_numpy(ep['depth_imgs'])

        if self.store:
            item['filename'] = ep_file

        return item


## Loads the simulator and episodes separately to enable per_scene collection of data
class HabitatDataScene(Dataset):

    def __init__(self, options, config_file, scene_id, existing_episode_list=[]):
        self.scene_id = scene_id

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cfg = habitat.get_config(config_file)
        cfg.defrost()
        #cfg.SIMULATOR.SCENE = '/media/ggeorgak/DATA/habitat_backup/data/scene_datasets/mp3d/' + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        cfg.SIMULATOR.SCENE = options.root_path + options.scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        #cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        cfg.SIMULATOR.TURN_ANGLE = options.turn_angle
        cfg.SIMULATOR.FORWARD_STEP_SIZE = options.forward_step_size
        cfg.freeze()

        self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        seed = 0
        self.sim.seed(seed)

        ## Load episodes of scene_id
        ep_file_path = options.root_path + options.episodes_root + cfg.DATASET.SPLIT + "/content/" + self.scene_id + ".json.gz"
        with gzip.open(ep_file_path, "rt") as fp:
            self.scene_data = json.load(fp)
        self.number_of_episodes = len(self.scene_data["episodes"])

        self.success_distance = cfg.TASK.SUCCESS.SUCCESS_DISTANCE

        ## Dataloader params
        self.hfov = float(cfg.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
        self.cfg_norm_depth = cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
        self.max_depth = cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.min_depth = cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        self.spatial_labels = options.n_spatial_classes
        self.object_labels = options.n_object_classes
        self.grid_dim = (options.grid_dim, options.grid_dim)
        self.cell_size = options.cell_size
        self.crop_size = (options.crop_size, options.crop_size)
        self.img_size = (options.img_size, options.img_size)
        self.img_segm_size = (options.img_segm_size, options.img_segm_size)
        self.normalize = True
        self.pixFormat = 'NCHW'
        
        #self.preprocessed_scenes_dir = "/media/ggeorgak/DATA/habitat_backup/data/scene_datasets/mp3d_scene_pclouds/"
        self.preprocessed_scenes_dir = options.root_path + options.scenes_dir + "mp3d_scene_pclouds/"

        self.episode_len = options.episode_len
        self.truncate_ep = options.truncate_ep

         # get point cloud and labels of scene
        self.pcloud, self.label_seq_spatial, self.label_seq_objects = utils.load_scene_pcloud(self.preprocessed_scenes_dir,
                                                                                                    self.scene_id, self.object_labels)
        self.color_pcloud = utils.load_scene_color(self.preprocessed_scenes_dir, self.scene_id)

        if len(existing_episode_list)!=0:
            self.existing_episode_list = [ int(x.split('_')[2]) for x in existing_episode_list ]
        else:
            self.existing_episode_list=[]

        self.occ_from_depth = options.occ_from_depth
        self.occupancy_height_thresh = options.occupancy_height_thresh

        # Build 3D transformation matrices
        self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_size[0]), np.linspace(1,-1,self.img_size[1])), device='cuda')
        self.xs = self.xs.reshape(1,self.img_size[0],self.img_size[1])
        self.ys = self.ys.reshape(1,self.img_size[0],self.img_size[1])
        K = np.array([
            [1 / np.tan(self.hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(self.hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.inv_K = torch.tensor(np.linalg.inv(K), device='cuda')
        # create the points2D containing all image coordinates
        x, y = torch.tensor(np.meshgrid(np.linspace(0, self.img_size[0]-1, self.img_size[0]), np.linspace(0, self.img_size[1]-1, self.img_size[1])), device='cuda')
        xy_img = torch.vstack((x.reshape(1,self.img_size[0],self.img_size[1]), y.reshape(1,self.img_size[0],self.img_size[1])))
        points2D_step = xy_img.reshape(2, -1)
        self.points2D_step = torch.transpose(points2D_step, 0, 1) # Npoints x 2

        self.scene = self.sim.semantic_annotations()


    def __len__(self):
        return self.number_of_episodes


    def __getitem__(self, idx):
        episode = self.scene_data['episodes'][idx]

        len_shortest_path = len(episode['shortest_paths'][0])
        objectgoal = episode['object_category']

        if len_shortest_path > 50: # skip that episode to avoid memory issues
            return None
        if len_shortest_path < self.episode_len+1:
            return None

        if idx in self.existing_episode_list:
            print("Episode", idx, 'already exists!')
            return None

        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in self.scene.objects}
        # convert the labels to the reduced set of categories
        instance_id_to_label_id_3 = instance_id_to_label_id.copy()
        instance_id_to_label_id_objects = instance_id_to_label_id.copy()
        for inst_id in instance_id_to_label_id.keys():
            curr_lbl = instance_id_to_label_id[inst_id]
            instance_id_to_label_id_3[inst_id] = viz_utils.label_conversion_40_3[curr_lbl]
            instance_id_to_label_id_objects[inst_id] = viz_utils.label_conversion_40_27[curr_lbl]

        # if truncated, run episode only up to the chosen step start_ind+episode_len
        if self.truncate_ep:
            start_ind = random.randint(0, len_shortest_path-self.episode_len-1)
            episode_extend = start_ind+self.episode_len
        else:
            episode_extend = len_shortest_path

        # imgs, depth, and ssegs stored here are (128,128) rather than the simulator's self.img_size:(256,256)
        # because they are going to be used during image segmentation training
        imgs = torch.zeros((episode_extend, 3, self.img_segm_size[0], self.img_segm_size[1]), dtype=torch.float32, device=self.device)
        depth_imgs = torch.zeros((episode_extend, 1, self.img_segm_size[0], self.img_segm_size[1]), dtype=torch.float32, device=self.device)
        ssegs_objects = torch.zeros((episode_extend, 1, self.img_segm_size[0], self.img_segm_size[1]), dtype=torch.float32, device=self.device)

        ssegs_3 = torch.zeros((episode_extend, 1, self.img_size[1], self.img_size[0]), dtype=torch.float32, device=self.device)

        points2D, local3D, abs_poses, rel_poses, action_seq, agent_height = [], [], [], [], [], []

        self.sim.reset()
        self.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
        sim_obs = self.sim.get_sensor_observations()
        observations = self.sim._sensor_suite.get_observations(sim_obs)


        for i in range(episode_extend):
            img = observations['rgb'][:,:,:3]
            depth_obsv = observations['depth'].permute(2,0,1).unsqueeze(0)

            depth = F.interpolate(depth_obsv.clone(), size=self.img_size, mode='nearest')
            depth = depth.squeeze(0).permute(1,2,0)

            if self.cfg_norm_depth:
                depth = utils.unnormalize_depth(depth, min=self.min_depth, max=self.max_depth)            

            semantic = observations['semantic']
            semantic = F.interpolate(semantic.unsqueeze(0).unsqueeze(0).float(), size=self.img_size, mode='nearest').int()
            semantic = semantic.squeeze(0).squeeze(0)

            # visual and 3d info
            imgData = utils.preprocess_img(img, cropSize=self.img_segm_size, pixFormat=self.pixFormat, normalize=self.normalize)
            local3D_step = utils.depth_to_3D(depth, self.img_size, self.xs, self.ys, self.inv_K)

            ssegData = np.expand_dims(semantic.cpu().numpy(), 0).astype(float) # 1 x H x W
            ssegData_3 = np.vectorize(instance_id_to_label_id_3.get)(ssegData.copy()) # convert instance ids to category ids
            ssegData_objects = np.vectorize(instance_id_to_label_id_objects.get)(ssegData.copy()) # convert instance ids to category ids

            agent_pose, y_height = utils.get_sim_location(agent_state=self.sim.get_agent_state())

            imgs[i,:,:,:] = imgData
            depth_resize = F.interpolate(depth_obsv.clone(), size=self.img_segm_size, mode='nearest')
            depth_imgs[i,:,:,:] = depth_resize.squeeze(0)
            ssegs_3[i,:,:,:] = torch.from_numpy(ssegData_3).float()
            ssegData_resize = F.interpolate(torch.from_numpy(ssegData_objects).unsqueeze(0).float(), size=self.img_segm_size, mode='nearest')
            ssegs_objects[i,:,:,:] = ssegData_resize.squeeze()

            abs_poses.append(agent_pose)
            agent_height.append(y_height)
            points2D.append(self.points2D_step)
            local3D.append(local3D_step)

            # get the relative pose with respect to the first pose in the sequence
            rel = utils.get_rel_pose(pos2=abs_poses[i], pos1=abs_poses[0])
            rel_poses.append(rel)

            # explicitly clear observation otherwise they will be kept in memory the whole time
            observations = None

            action_id = episode['shortest_paths'][0][i]
            if action_id==None:
                break
            observations = self.sim.step(action_id)


        pose = torch.from_numpy(np.asarray(rel_poses)).float()
        abs_pose = torch.from_numpy(np.asarray(abs_poses)).float()

        # Create the ground-projected grids
        if self.occ_from_depth:
            ego_grid_sseg_3 = map_utils.est_occ_from_depth(local3D, grid_dim=self.grid_dim, cell_size=self.cell_size, 
                                                    device=self.device, occupancy_height_thresh=self.occupancy_height_thresh)
        else:
            ego_grid_sseg_3 = map_utils.ground_projection(self.points2D, local3D, ssegs_3, sseg_labels=self.spatial_labels, grid_dim=self.grid_dim, cell_size=self.cell_size)

        ego_grid_crops_3 = map_utils.crop_grid(grid=ego_grid_sseg_3, crop_size=self.crop_size)
        step_ego_grid_3 = map_utils.get_acc_proj_grid(ego_grid_sseg_3, pose, abs_pose, self.crop_size, self.cell_size)
        step_ego_grid_crops_3 = map_utils.crop_grid(grid=step_ego_grid_3, crop_size=self.crop_size)
        # Get cropped gt
        gt_grid_crops_spatial = map_utils.get_gt_crops(abs_pose, self.pcloud, self.label_seq_spatial, agent_height,
                                                            self.grid_dim, self.crop_size, self.cell_size)
        gt_grid_crops_objects = map_utils.get_gt_crops(abs_pose, self.pcloud, self.label_seq_objects, agent_height,
                                                            self.grid_dim, self.crop_size, self.cell_size)

        item = {}
        item['images'] = imgs
        item['depth_imgs'] = depth_imgs
        item['ssegs'] = ssegs_objects
        item['episode_id'] = idx
        item['scene_id'] = self.scene_id
        item['abs_pose'] = abs_pose
        item['ego_grid_crops_spatial'] = ego_grid_crops_3
        item['step_ego_grid_crops_spatial'] = step_ego_grid_crops_3
        item['gt_grid_crops_spatial'] = gt_grid_crops_spatial
        item['gt_grid_crops_objects'] = gt_grid_crops_objects
        return item



class HabitatDataRxR(Dataset):

    # Loads the precomputed maps and the RxR episodes
    # Each training example should contain:
    #   1) Map tensor
    #   2) Either BERT features or the instruction itself
    #   3) Waypoint heatmaps  

    def __init__(self, options, eval_set):
        #self.scene_id = scene_id
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = options
        self.is_train = options.is_train
        self.token_inputs = options.token_inputs

        # use specific split for now: val_unseen
        # need to filter the data for the english instructions
        self.episodes_list = [] # each element is a dict holding relevant episode info

        split = options.split #'train' #'train' #'val_unseen' #'val_seen'

        if split=="val_unseen" and options.cell_size==0.1:
            self.map_dir = options.root_map_dir + options.scenes_dir + "mp3d_maps/val_unseen_old/"
        else:
            self.map_dir = options.root_map_dir + options.scenes_dir + "mp3d_maps/" + split + "/"

        ########################### read rxr data
        root_rxr_dir = options.root_path + "rxr-data/"
        #root_rxr_dir = '/media/ggeorgak/DATA/rxr-data/'
        self.pose_traces_dir = root_rxr_dir + "pose_traces/rxr_"+split+"/"
        self.text_features_dir = root_rxr_dir + "text_features/rxr_"+split+"/"

        follower_file = root_rxr_dir + "rxr_"+split+"_follower.jsonl.gz"
        guide_file = root_rxr_dir + "rxr_"+split+"_guide.jsonl.gz"

        guide_data = utils.read_json_lines(guide_file)
        follower_data = utils.read_json_lines(follower_file)

        #print(len(guide_data))
        #print(guide_data[36])
        #print(len(follower_data))
        #print(follower_data[2])

        ## * We should be using the RxR_VLNCE_v0 episodes when training for the actual VLN-CE task because they contain goals and paths
        ## * Need to verify that the pose traces from RxR have correspondence to episodes in RxR_VLNCE (using the instruction_id)
        ## * RxR_VLNCE_v0 is downloaded from: https://github.com/jacobkrantz/VLN-CE/tree/rxr-habitat-challenge 
        ## * R2R_VLNCE_v1-2 is downloaded from: https://github.com/jacobkrantz/VLN-CE (probably does not contain pose traces). These are the old R2R episodes converted to continuous environments 
        #tmp_data = utils.read_json_lines('/media/ggeorgak/DATA/rxr-data/RxR_VLNCE_v0/train/train_guide.json.gz') # contains similar episodes to RxR with specific goal position
        #tmp_data = utils.read_json_lines('/media/ggeorgak/DATA/rxr-data/RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz') # contains similar episodes to RxR with specific goal position
        #tmp_data_gt = utils.read_json_lines('/media/ggeorgak/DATA/rxr-data/RxR_VLNCE_v0/val_unseen/val_unseen_guide_gt.json.gz') # contains ground-truth path and actions
        #print(tmp_data[0].keys())
        #print(tmp_data[0]['episodes'][0].keys())
        #print(tmp_data[0]['episodes'][0]['instruction'])
        #print(len(tmp_data[0]['episodes']))
        #print()
        #print(tmp_data_gt[0]['1'])
        

        ## ** for the case of solving the actual VLN:
        ## Use this dataloader and write a new model that takes in the map, the start location, and the instruction, and predicts the goal (or maybe the waypoints also)
        ## Need to create the gt map for each example using the initial position of the agent.
        ## Should the predicted goal be as orientation difference and distance (relative to the current agent pose?)
        ## Think of the actual execution of the task, what map (or input) should we pass at every time-step? i.e. global, cropped, oriented in what way?
        ## What (map, instruction, pose) --> goal problem should we train for?

        # filter episodes based on the list of available mp3d maps of split
        available_maps = os.listdir(self.map_dir)
        #print(available_maps)

        #scene_list=[]

        invalid_instruction_ids = utils.get_invalid_ids(split)

        for i in range(len(guide_data)):
            guide_annot = guide_data[i]
            follower_annot = follower_data[i]
            info = {}
            if guide_annot['language'] == "en-IN" or guide_annot['language'] == "en-US":
                
                if guide_annot['instruction_id'] in invalid_instruction_ids:
                    continue
                
                if guide_annot['scan'] not in available_maps:
                    continue
                
                #if guide_annot['scan'] not in scene_list:
                #    scene_list.append(guide_annot['scan'])

                info['path_id'] = guide_annot['path_id']
                info['scene_id'] = guide_annot['scan']
                info['instruction_id'] = guide_annot['instruction_id']
                info['instruction'] = guide_annot['instruction']
                info['demonstration_id'] = follower_annot['demonstration_id']
                info['path'] = follower_annot['path']

                #info['metrics'] = follower_annot['metrics']
                self.episodes_list.append(info)
        print("Total data:", len(self.episodes_list))
        #scene_list.sort()
        #print(scene_list)
        #print(len(scene_list))

        '''
        # ** check for empty poses valid
        inv_instr=[]
        for i in range(len(self.episodes_list)):
            print(i)
            episode = self.episodes_list[i]
            pose_trace_path = self.pose_traces_dir + f'{episode["instruction_id"]:06}_guide_pose_trace.npz'
            poses_traced = np.load(pose_trace_path)["extrinsic_matrix"]
            poses_valid, heights = utils.filter_pose_trace(poses_traced[::10])

            if len(poses_valid)<2:
                print(i)
                print("Pose trace len:", len(poses_traced))
                print("Pose valid len:", len(poses_valid))
                inv_instr.append(episode["instruction_id"])
                #raise Exception('Poses valid of instruction', episode["instruction_id"], "is empty!")

            waypoints = utils.sample_waypoints(poses_valid, num_waypoints=options.num_waypoints) # waypoints in x,y,o habitat coordinate space
            if waypoints==None:
                print('Waypoints less than 2!')
                inv_instr.append(episode["instruction_id"])
            
            if i % 100 == 0:
                print("Invalid instruction ids:", inv_instr)

        print(inv_instr)
        '''

        self.grid_dim = (options.grid_dim, options.grid_dim) # ** this needs to match the size of the stored map
        self.sg = SemanticGrid(1, self.grid_dim, options.crop_size, options.cell_size,
                                    spatial_labels=options.n_spatial_classes, object_labels=options.n_object_classes)

        self.num_waypoints = options.num_waypoints
        self.heatmap_size = (options.heatmap_size, options.heatmap_size)

        # Do train-test split to ensure that all scenes are observed during training
        # dictionary for scene occurences
        scenes={}
        for i in range(len(self.episodes_list)):
            info = self.episodes_list[i]
            if info['scene_id'] not in scenes:
                scenes[info['scene_id']] = [] #0
            scenes[info['scene_id']].append(i)  #+= 1
        #print(scenes)

        train_set, val_set = [], []
        for key in scenes.keys():
            idxs = scenes[key]
            cut = int(len(idxs)*0.95)
            train_set += idxs[:cut]
            val_set += idxs[cut:]
            #print(cut, len(train_set), len(val_set))

        if not eval_set:
            #self.episodes_list = self.episodes_list[:-500]
            self.episodes_idx = train_set
            #self.episodes_idx = train_set[:20]
        else:
            #self.episodes_list = self.episodes_list[-500:]
            #self.episodes_list = self.episodes_list[:10]
            self.episodes_idx = val_set


        
    def __len__(self):
        #return len(self.episodes_list)
        return len(self.episodes_idx)


    def __getitem__(self, idx):
        ind = self.episodes_idx[idx] # get the next episode index
        #ind = 0 # **** hardcode the ind for now for debugging
        episode = self.episodes_list[ind]

        #print("idx:", ind)
        #print(episode)

        # Given info from episode, load the map, prepare the keypoints (pose traces), and load the BERT features
        # use either instruction_id or demonstration_id to load the pose traces and the features
        map_info_path = self.map_dir + episode['scene_id'] + "/map_info.npz"
        map_info = np.load(map_info_path) # Keys: maps_semantic, maps_occupancy, top_levels_heights, origin_pose

        maps_semantic = torch.tensor(map_info['maps_semantic']) # holds maps for all levels

        if 'cell_size' in map_info.keys(): # case of the newely generated maps with cell_size=0.05
            assert map_info['cell_size']==self.options.cell_size, 'Cell size needs to be equivalent to the map cell_size!'

        #print(maps_semantic.shape)
        levels_heights = map_info['top_levels_heights']
        origin_pose = torch.tensor(map_info['origin_pose']).float()

        # For now assume we will use the guide annotations. If we switch to follower, then
        # we need to filter them out based on the success rate of the follower.
        #pose_trace_path = self.pose_traces_dir + f'{episode["demonstration_id"]:06}_follower_pose_trace.npz'
        pose_trace_path = self.pose_traces_dir + f'{episode["instruction_id"]:06}_guide_pose_trace.npz'
        poses_traced = np.load(pose_trace_path)["extrinsic_matrix"]

        if len(poses_traced)==0:
            raise Exception('Poses traced of instruction', episode["instruction_id"], "is empty!")

        # Sample the poses to create the waypoints sequence
        poses_valid, heights = utils.filter_pose_trace(poses_traced[::10])
        
        if len(poses_valid)<2:
            print("Pose trace len:", len(poses_traced))
            print("Pose valid len:", len(poses_valid))
            raise Exception('Poses valid of instruction', episode["instruction_id"], "is empty!")

        waypoints = utils.sample_waypoints(poses_valid, num_waypoints=self.num_waypoints) # waypoints in x,y,o habitat coordinate space

        if waypoints.shape[0] != self.num_waypoints:
            raise Exception('Number of waypoints '+str(waypoints.shape[0])+" is not correct!")

        # Transform waypoints to map coordinates
        pose_coords = torch.zeros((len(waypoints), 2))
        for i in range(len(waypoints)):
            rel_pose = utils.get_rel_pose(pos2=waypoints[i], pos1=origin_pose)
            _rel_pose = torch.Tensor(rel_pose).unsqueeze(0).float()
            pose_coords0 = tutils.get_coord_pose(self.sg, _rel_pose, origin_pose, self.grid_dim[0], self.options.cell_size) # B x T x 2
            pose_coords0 = pose_coords0.squeeze(0)
            pose_coords[i,:] = pose_coords0.squeeze(0).clone()

        # The pose traces have multiple headings at each location, so during sampling it is very likely we will sample a 
        # heading that does not correspond to the trajectory direction or the instructions (turn left, etc).
        # One way to deal with this is if do atan2() between the sampled waypoints, and just use the first true heading for the first waypoint.
        # Convention: heading 0 is looking upwards, + right, - left. Add pi to true first heading and pi/2 to the rest to conform to the convention. 
        angle_headings = torch.zeros((len(pose_coords)))
        angle_headings[0] = utils.wrap_angle(waypoints[0,2] + np.pi)
        for i in range(pose_coords.shape[0]-1):
            p0, p1 = pose_coords[i], pose_coords[i+1]
            angle = torch.atan2(torch.tensor([p1[1]-p0[1]]), torch.tensor([p1[0]-p0[0]]))
            angle_headings[i+1] = utils.wrap_angle(angle + np.pi/2)
        #print(angle_headings.shape)
        #print(angle_headings)

        # We learn to predict the headings in terms of cos(phi) and sin(phi)
        # The angle can be recovered by: phi = torch.atan2(sin(phi), cos(phi))
        cos_headings = torch.cos(angle_headings)
        sin_headings = torch.sin(angle_headings)
        #print(cos_headings)
        #print(sin_headings)
        #print(torch.atan2(sin_headings, cos_headings))
        headings = torch.stack((sin_headings, cos_headings)) # 2 x 10
        #print(headings.shape)

        # Convert the waypoints to keypoint heatmaps
        keypoints_heat = utils.locs_to_heatmaps(keypoints=pose_coords, img_size=self.grid_dim, out_size=self.heatmap_size)
        #print(keypoints_heat.shape)

        # Load the BERT instruction features
        bert_feat_path = self.text_features_dir + f'{episode["instruction_id"]:06}_en_text_features.npz'
        bert_feat = np.load(bert_feat_path)
        #print(list(bert_feat.keys()))
        #print("Bert feat:", bert_feat['features'].shape)

        # do zero-padding of features
        #text_feat = torch.zeros((self.max_len, 768)) #, device=self.device)
        #text_feat[:bert_feat['features'].shape[0], :] = torch.from_numpy(bert_feat['features'])

        # Choose which level to use from the map, based on the heights of the pose trace
        if len(levels_heights) > 1:
            #mean_height = np.mean(np.asarray(heights))
            median_height = np.median(np.asarray(heights))
            dist = abs(levels_heights-median_height)
            level_idx = torch.argmin(torch.tensor(dist)).item()
        else:
            level_idx = 0

        #viz_utils.vis_episode(gt_map_semantic=maps_semantic[level_idx], pose_coords=pose_coords, name=str(idx)+"_"+episode['scene_id'])

        # Add noise to the map 
        map_gt = maps_semantic[level_idx]
        map_gt_noisy = utils.add_uniform_noise(tensor=map_gt.clone(), a=-0.2, b=0.2)

        item = {}

        if self.options.with_rgb_maps:
            # add noise to map color
            maps_color = torch.tensor(map_info['maps_color'])
            map_color = maps_color[level_idx]
            map_color_noisy = utils.add_uniform_noise(tensor=map_color.clone(), a=-0.2, b=0.2)
            item['map_color'] = map_color_noisy
        
        item['map_semantic'] = map_gt_noisy #maps_semantic[level_idx]
        item['text_feat'] = torch.from_numpy(bert_feat['features'])
        #item['text_feat'] = text_feat #torch.rand(random.randint(100, 180), 768)
        item['keypoint_heatmaps'] = keypoints_heat
        item['headings'] = headings
        item['angle_headings'] = angle_headings
        item['map_semantic_noise_free'] = map_gt

        # during testing return the instruction and the token correspondence
        if not self.is_train:
            item['instruction'] = episode['instruction']
            item['tokens'] = bert_feat['tokens']

            # if chosen, overwrite the text_feat 
            if self.token_inputs=="shuffle":
                feat = torch.from_numpy(bert_feat['features'])
                rand_idx = torch.randperm(feat.size()[0])
                feat = feat[rand_idx] # shuffles across rows
                tokens = bert_feat['tokens'][rand_idx] # need to shuffle the tokens as well
                item['text_feat'] = feat
                item['tokens'] = tokens
            elif self.token_inputs=="garbage":
                item['text_feat'] = torch.randn(bert_feat['features'].shape)
                
        return item



class HabitatDataVLN(Dataset):

    # Loads necessary data for the actual VLN task

    def __init__(self, options, config_file, scene_id, existing_episode_list=[], random_poses=False, pose_noise=1):

        self.options = options
        self.scene_id = scene_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_poses_per_example = options.num_poses_per_example

        #sets = ['R2R_VLNCE_v1-2', 'rxr-data'] #['R2R_VLNCE_v1-2'] #['rxr-data']
        self.parse_episodes(self.options.datasets)
        
        self.number_of_episodes = len(self.scene_data["episodes"])

        #print(self.number_of_episodes)
        #print(self.scene_data['episodes'][0])

        cfg = habitat.get_config(config_file)
        cfg.defrost()
        #cfg.SIMULATOR.SCENE = '/media/ggeorgak/DATA/habitat_backup/data/scene_datasets/mp3d/' + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        cfg.SIMULATOR.SCENE = options.root_path + options.scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        #cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        cfg.SIMULATOR.TURN_ANGLE = options.turn_angle
        cfg.SIMULATOR.FORWARD_STEP_SIZE = options.forward_step_size
        cfg.freeze()

        self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        seed = 0
        self.sim.seed(seed)

        self.cfg_norm_depth = cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
        self.object_labels = options.n_object_classes
        self.grid_dim = (options.grid_dim, options.grid_dim)
        self.global_dim = (options.global_dim, options.global_dim)
        self.cell_size = options.cell_size
        self.heatmap_size = (options.heatmap_size, options.heatmap_size)
        self.num_waypoints = options.num_waypoints
        self.min_angle_noise = np.radians(-15)
        self.max_angle_noise = np.radians(15)
        self.img_size = (options.img_size, options.img_size)
        #self.crop_size = (options.crop_size, options.crop_size)
        self.max_depth = cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.min_depth = cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH

        #self.preprocessed_scenes_dir = "/media/ggeorgak/DATA/habitat_backup/data/scene_datasets/mp3d_scene_pclouds/"
        self.preprocessed_scenes_dir = options.root_path + options.scenes_dir + "mp3d_scene_pclouds/"

         # get point cloud and labels of scene
        self.pcloud, self.label_seq_spatial, self.label_seq_objects = utils.load_scene_pcloud(self.preprocessed_scenes_dir,
                                                                                                    self.scene_id, self.object_labels)
        self.color_pcloud = utils.load_scene_color(self.preprocessed_scenes_dir, self.scene_id)

        # Initialize the semantic grid only to use the spatialTransformer. The crop_size (heatmap_size) argument does not matter here
        self.sg = SemanticGrid(1, self.grid_dim, options.heatmap_size, self.cell_size,
                                    spatial_labels=options.n_spatial_classes, object_labels=options.n_object_classes)
        #self.sg_global = SemanticGrid(1, self.global_dim, options.heatmap_size, self.cell_size,
        #                            spatial_labels=options.n_spatial_classes, object_labels=options.n_object_classes)

        if len(existing_episode_list)!=0:
            self.existing_episode_list = [ int(x.split('_')[2]) for x in existing_episode_list ]
        else:
            self.existing_episode_list=[]
        
        self.random_poses = random_poses
        self.pose_noise = pose_noise # used during store_vln episodes


    def parse_episodes(self, sets):
        # Need to merge all different sets (rxr-data, R2R_VLNCE, RxR_VLNCE) for the training
        # Load episodes from multiple sources and convert to a common format (start_pos, goal, instruction)

        self.scene_data = {'episodes': []}

        for s in sets:

            if s=='R2R_VLNCE_v1-2':
                root_rxr_dir = self.options.root_path + "rxr-data/" + s + "/"
                episode_file = root_rxr_dir + self.options.split + "/" + self.options.split + ".json.gz"
                with gzip.open(episode_file, "rt") as fp:
                    self.data = json.load(fp)

                '''
                # count the number of episodes per scene
                scene_count={}
                dist_list = []
                for i in range(len(self.data['episodes'])):
                    sc_path = self.data['episodes'][i]['scene_id']
                    sc_id = sc_path.split('/')[-1].split('.')[0]
                    start_pos = self.data['episodes'][i]['start_position']
                    goal_pos = self.data['episodes'][i]['goals'][0]['position']
                    if np.absolute(start_pos[1] - goal_pos[1]) >= 0.2:
                        continue
                    ltg_dist = torch.linalg.norm(torch.tensor(start_pos)-torch.tensor(goal_pos))
                    dist_list.append(ltg_dist.item())
                    if sc_id not in scene_count:
                        scene_count[sc_id] = 1
                    else:
                        scene_count[sc_id] += 1
                #print(scene_count)
                count_all=0
                sc_list = list(scene_count.keys())
                sc_list.sort()
                for sc in sc_list:
                    print(sc, scene_count[sc])
                    count_all+=scene_count[sc]
                #for sc in scene_count.keys():
                #    print(sc, scene_count[sc])
                #    count_all+=scene_count[sc]
                print(len(sc_list))
                print("All:", count_all)
                print(dist_list)
                print(np.mean(np.asarray(dist_list)))
                with open('val_seen_dist.npy', 'wb') as f:
                    np.save(f, dist_list)
                '''

                if self.options.split!="test":
                    # Load the gt information from R2R_VLNCE
                    episode_file_gt = self.options.root_path+"rxr-data/"+s+"_preprocessed/"+self.options.split +"/"+self.options.split+"_gt.json.gz"
                    with gzip.open(episode_file_gt, "rt") as fp:
                        self.data_gt = json.load(fp)
                
                #print(self.data.keys())
                #self.instruction_vocab = self.data['instruction_vocab']
                # Need to keep only episodes that belong to current scene
                for i in range(len(self.data['episodes'])):
                    sc_path = self.data['episodes'][i]['scene_id']
                    sc_id = sc_path.split('/')[-1].split('.')[0]
                    if sc_id == self.scene_id:                        
                        # Check if given path has enough poses
                        #ref_path = self.data['episodes'][i]['reference_path']
                        #if len(ref_path) < self.num_poses_per_example:
                        #    continue

                        # seems that the "start_rotation" for the R2R_VLNCE_v1-2 set has 180 degrees difference from the description
                        #self.data['episodes'][i]['start_rotation'] = utils.add_to_quarternion(rotation=self.data['episodes'][i]['start_rotation'], angle=-np.pi)
                        self.data['episodes'][i]['scene_id'] = self.scene_id
                        self.data['episodes'][i]['dataset'] = s
                        
                        if self.options.split!="test":
                            # check whether goal is at the same height as start position
                            start_pos = self.data['episodes'][i]['start_position']
                            goal_pos = self.data['episodes'][i]['goals'][0]['position']
                            if np.absolute(start_pos[1] - goal_pos[1]) >= 0.2 and self.options.check_floor:
                                continue
                            # get gt info
                            gt_info = self.data_gt[ str(self.data['episodes'][i]['episode_id']) ] # locations, forward_steps, actions
                            #print(len(gt_info['locations']), gt_info['forward_steps'], len(gt_info['actions']))
                            self.data['episodes'][i]['waypoints'] = gt_info['locations']
                            self.data['episodes'][i]['actions'] = gt_info['actions']
                            

                        self.scene_data['episodes'].append(self.data['episodes'][i])
                            

            elif s=="rxr-data":
                # use the same format as the R2R_VLNCE, don't use text_embeddings, we should define a model for that
                root_rxr_dir = self.options.root_path + "rxr-data/"
                self.pose_traces_dir = root_rxr_dir + "pose_traces/rxr_"+self.options.split+"/"
                #self.text_features_dir = root_rxr_dir + "text_features/rxr_"+self.options.split+"/"
                guide_file = root_rxr_dir + "rxr_"+self.options.split+"_guide.jsonl.gz"
                guide_data = utils.read_json_lines(guide_file)
                # the pre-saved maps here are used to get the level heights (until we figure out a better way to do it)
                self.map_dir = self.options.root_map_dir + self.options.scenes_dir + "mp3d_maps/" + self.options.split + "/"
                #self.map_dir = self.options.root_map_dir + "mp3d_maps/" + self.options.split + "/"
                map_info_path = self.map_dir + self.scene_id + "/map_info.npz"
                levels_heights = np.load(map_info_path)['top_levels_heights'] # Keys: maps_semantic, maps_occupancy, top_levels_heights, origin_pose
                invalid_instruction_ids = utils.get_invalid_ids(self.options.split)

                '''
                # count episodes per scene
                scene_count={}
                for i in range(len(guide_data)):
                    guide_annot = guide_data[i]
                    if guide_annot['language'] == "en-IN" or guide_annot['language'] == "en-US":                    
                        if guide_annot['instruction_id'] in invalid_instruction_ids:
                            continue
                        if guide_annot['scan'] not in scene_count:
                            scene_count[guide_annot['scan']] = 1
                        else:
                            scene_count[guide_annot['scan']] += 1
                count_all=0
                sc_list = list(scene_count.keys())
                sc_list.sort()
                for sc in sc_list:
                    print(sc, scene_count[sc])
                    count_all+=scene_count[sc]
                print(len(sc_list))
                print("All:", count_all)
                raise Exception('777')
                '''
                
                for i in range(len(guide_data)):
                    guide_annot = guide_data[i]
                    info = {}
                    if guide_annot['language'] == "en-IN" or guide_annot['language'] == "en-US":
                        
                        if guide_annot['instruction_id'] in invalid_instruction_ids:
                            continue
                        
                        if guide_annot['scan'] == self.scene_id:
                        
                            info['dataset'] = s # 'rxr-data'
                            info['path_id'] = guide_annot['path_id']
                            info['scene_id'] = guide_annot['scan']
                            info['instruction_id'] = guide_annot['instruction_id']
                            info['episode_id'] = i

                            # find start pose and goal pos from pose_trace
                            pose_trace_path = self.pose_traces_dir + f'{guide_annot["instruction_id"]:06}_guide_pose_trace.npz'
                            poses_traced = np.load(pose_trace_path)["extrinsic_matrix"]
                            init_pose = poses_traced[0]
                            last_pose = poses_traced[-1]

                            # check whether goal is at the same height as start position
                            if np.absolute(init_pose[1,3] - last_pose[1,3]) > 0.2:
                                continue

                            info['start_position'], info['start_rotation'] = utils.get_episode_pose(init_pose)
                            #info['poses_traced'] = poses_traced
                            waypoints = []
                            poses_traced = poses_traced[::10]
                            for k in range(poses_traced.shape[0]):
                                pos, _ = utils.get_episode_pose(poses_traced[k])
                                waypoints.append(pos.tolist())

                            info['waypoints'] = waypoints
                            goal_position, _ = utils.get_episode_pose(last_pose)
                            info['goals'] = []
                            info['goals'].append( {'position': goal_position} )

                            # get heights from pose trace to determine the slice
                            heights = poses_traced[:,1,3]
                            if len(levels_heights) > 1:
                                median_height = np.median(np.asarray(heights))
                                dist = abs(levels_heights-median_height)
                                level_idx = torch.argmin(torch.tensor(dist)).item()
                            else:
                                level_idx = 0
                            # replace height in start position with found level height
                            info['start_position'][1] = levels_heights[level_idx]
                            info['goals'][0]['position'][1] = levels_heights[level_idx]
                            info['level_height'] = levels_heights[level_idx]

                            # instruction related info 
                            #bert_feat_path = self.text_features_dir + f'{guide_annot["instruction_id"]:06}_en_text_features.npz'
                            #bert_feat = np.load(bert_feat_path)
                            #print(list(bert_feat.keys()))
                            info['instruction'] = {'instruction_text': guide_annot['instruction'],
                                                   #'instruction_tokens': bert_feat['tokens'],
                                                   #'instruction_features': bert_feat['features'] 
                                                   }
                            # ** Probably need to predict goals sequentially from segments of the instruction
                            # ** Consider predicting distance and orientation instead of keypoint heatmap 
                            self.scene_data['episodes'].append(info)


    def __len__(self):
        return self.number_of_episodes

    '''
    def transform_to_map_coords(self, position, abs_pose, grid_size):
        pose, _ = tutils.get_2d_pose(position=position)
        agent_rel_pose = utils.get_rel_pose(pos2=pose, pos1=abs_pose)
        agent_rel_pose = torch.Tensor(agent_rel_pose).unsqueeze(0).float()
        agent_rel_pose = agent_rel_pose.to(self.device)
        #_pose_coords = tutils.get_coord_pose(self.sg, agent_rel_pose, abs_pose, self.grid_dim[0], self.cell_size, self.device) # B x T x 3
        _pose_coords = tutils.get_coord_pose(self.sg, agent_rel_pose, abs_pose, grid_size, self.cell_size, self.device) # B x T x 3

        visible_position = 1
        # if goal pose coords is 0,0 then goal is outside the current map. Use an empty heatmap
        if _pose_coords[0,0,0]==0 and _pose_coords[0,0,1]==0:
            _pose_coords = torch.tensor([[[-200,-200]]])
            visible_position = 0
        return _pose_coords, visible_position
    '''

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


    def __getitem__(self, idx):
        #idx = 11 # ** tmp hardcode
        #print("Episode", idx)
        
        #if idx in self.existing_episode_list:
        #    print("Episode", idx, 'already exists!')
        #    return None
        
        episode = self.scene_data['episodes'][idx]
        #print(episode)

        instruction = episode['instruction']['instruction_text']
        #print(instruction)
        
        if episode['dataset'] == 'rxr-data':
            #bert_feat_path = self.text_features_dir + f'{episode["instruction_id"]:06}_en_text_features.npz'
            #bert_feat = np.load(bert_feat_path)
            #instruction_feat = bert_feat['features']
            #instruction_tokens = bert_feat['tokens']
            #print(instruction_tokens)
            goal_position = episode['goals'][0]['position']
            # get valid map locations to set the sim from poses_traced
            poses_traced = episode['poses_traced']
            #print(poses_traced.shape)
            _, _, inds = utils.filter_pose_trace(poses_traced, return_idx=True)
            # get the valid poses_traced
            poses_traced = poses_traced[inds]

            if len(poses_traced) < self.num_poses_per_example+1:
                return None

            poses_traced = poses_traced[:-1] # remove last pose (goal)
            #print(poses_traced.shape)
            # randomly select a constant number of poses 
            chosen_inds = random.sample(range(1,poses_traced.shape[0]), k=self.num_poses_per_example-1)
            chosen_inds.append(0) # keep start pose
            chosen_inds.sort()
            
            pose_positions, pose_rotations = [], []
            #for i in range(poses_traced.shape[0]):
            for i in range(len(chosen_inds)):
                pose_ind = chosen_inds[i]
                pos, rot = utils.get_episode_pose(poses_traced[pose_ind])
                pos[1] = episode['level_height']
                pose_positions.append(pos)
                # add small perturbation on the rotation to create diverse examples
                #rot = utils.add_to_quarternion(rotation=rot, angle=self.min_angle_noise + (random.random() * (self.max_angle_noise-self.min_angle_noise)))
                pose_rotations.append(rot)
        
        elif episode['dataset'] == 'R2R_VLNCE_v1-2':
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

        # set simulator pose at episode start
        self.sim.reset()
        self.sim.set_agent_state(episode["start_position"], episode["start_rotation"])

        # To sample locations with noise, randomly select 10 locations from episode['waypoints']
        # and randomly select orientation and add noise to the position. Move the simulator directly to those locations
        if self.random_poses:
            sim_positions, sim_rotations = self.sample_random_poses(episode)
            iterations = len(sim_positions)
        else:
            iterations = len(actions)

        abs_poses = []
        #goal_rel_pose_coords = torch.zeros((len(actions), 2), dtype=torch.float32, device=self.device)
        gt_maps = torch.zeros((iterations, 1, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        goal_maps = torch.zeros((iterations, self.num_waypoints, self.heatmap_size[0], self.heatmap_size[1]), dtype=torch.float32, device=self.device)
        visible_waypoints = torch.zeros((iterations, self.num_waypoints))
        covered_waypoints = torch.zeros((iterations, self.num_waypoints))

        ### Get egocentric map at each waypoint location along with its corresponding relative goal
        for t in range(iterations):

            if self.random_poses:
                self.sim.set_agent_state(sim_positions[t], sim_rotations[t])

            agent_pose, y_height = utils.get_sim_location(agent_state=self.sim.get_agent_state())
            abs_poses.append(agent_pose)

            # get gt map from agent pose (pose is at the center looking upwards)
            x, y, z, label_seq, color_pcloud = map_utils.slice_scene(x=self.pcloud[0].copy(),
                                                                y=self.pcloud[1].copy(),
                                                                z=self.pcloud[2].copy(),
                                                                label_seq=self.label_seq_objects.copy(),
                                                                height=y_height,
                                                                color_pcloud=self.color_pcloud)

            gt_map_semantic, gt_map_color = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[t],
                                                        grid_dim=self.grid_dim, cell_size=self.cell_size, color_pcloud=color_pcloud, z=z)
            #print(gt_map_semantic.shape)
            #print(gt_map_color.shape)
            #viz_utils.write_tensor_imgSegm(img=gt_map_semantic.unsqueeze(0).cpu(), savepath="", name="gt_map_semantic"+str(t))
            ##viz_utils.write_tensor_imgSegm(img=maps_occupancy.cpu(), savepath=save_dir, name="gt_map_occupancy", labels=3)
            #viz_utils.write_img(img=gt_map_color.unsqueeze(0).cpu(), savepath="", name="gt_map_color"+str(t))
            
            gt_maps[t,:,:,:] = gt_map_semantic

            # get the relative pose with respect to the first pose in the sequence
            rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[t])
            _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
            _rel_pose = _rel_pose.to(self.device)
            pose_coords = tutils.get_coord_pose(self.sg, _rel_pose, abs_poses[t], self.grid_dim[0], self.cell_size, self.device) # B x T x 3
            #print(pose_coords) # should always be in the middle of the map

            # Transform waypoints with respect to agent current pose
            waypoints_pose_coords = torch.zeros((len(waypoints), 2))
            for k in range(len(waypoints)):
                #point_pose_coords, visible = self.transform_to_map_coords(position=waypoints[k], abs_pose=abs_poses[t], grid_size=self.grid_dim[0])
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
            #viz_utils.vis_heatmaps(goal_maps[t,9,:,:], goal_maps[t,9,:,:])
            #viz_utils.save_map_goal(gt_maps[t,:,:,:].unsqueeze(0).unsqueeze(0), pose_coords, waypoints_pose_coords[-1].unsqueeze(0).unsqueeze(0), "", t)

            #goal_rel_pose_coords[t,:] = goal_pose_coords.squeeze(1)
            #print(goal_rel_pose_coords)

            if not self.random_poses:
                action_id = actions[t]
                self.sim.step(action_id)
        
        abs_poses = torch.from_numpy(np.asarray(abs_poses)).float()    

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
            gt_maps = gt_maps[inds, :, :, :] # num_poses x 1 x 192 x 192
            visible_waypoints = visible_waypoints[inds, :] # num_poses x num_waypoints
            covered_waypoints = covered_waypoints[inds, :] # num_poses x num_waypoints

        item = {}
        item['goal_heatmap'] = goal_maps #goal_heatmap
        item['map_semantic'] = gt_maps #gt_map_semantic.cpu()
        #item['tokens'] = torch.tensor(tokens) # list of word tokens corresponding to vocabulary
        #item['text_feat'] = torch.from_numpy(instruction_feat)
        item['abs_pose'] = abs_poses
        item['goal_position'] = goal_position # absolute goal position, consistent within an episode
        #item['goal_rel_pose_coords'] = goal_rel_pose_coords # relative map coords for each goal (agent is at [255,255])
        item['instruction'] = instruction #episode['instruction']['instruction_text']
        item['visible_waypoints'] = visible_waypoints
        item['covered_waypoints'] = covered_waypoints

        item['dataset'] = episode['dataset']
        item['episode_id'] = episode['episode_id']
        
        return item



class HabitatDataVLNOffline(Dataset):
    # Loads stored episodes for the VLN task

    def __init__(self, options, eval_set, use_all=False, offline_eval=False):
        #config = get_config(config_file)
        #self.config = config

        self.episodes_file_list = []
        self.episodes_file_list += self.collect_stored_episodes(options, split=options.split)
        #print(self.episodes_file_list)
        self.vln_no_map = options.vln_no_map
        self.vln_L2M = options.vln_L2M
        self.vln_no_map_ext = options.vln_no_map_ext
        self.use_all = use_all
        self.offline_eval = offline_eval
        self.sample_1 = options.sample_1

        if options.dataset_percentage < 1: # Randomly choose the subset of the dataset to be used
            random.shuffle(self.episodes_file_list)
            self.episodes_file_list = self.episodes_file_list[ :int(len(self.episodes_file_list)*options.dataset_percentage) ]


        if self.use_all: # no train/eval split
            self.episodes_idx = list(range(len(self.episodes_file_list)))
        else:
            # Do train-test split to ensure that all scenes are observed during training
            # dictionary for scene occurences
            scenes={}
            for i in range(len(self.episodes_file_list)):
                ep_path = self.episodes_file_list[i]
                scene_id = ep_path.split('/')[-2]            
                if scene_id not in scenes:
                    scenes[scene_id] = [] #0
                scenes[scene_id].append(i)  #+= 1

            train_set, val_set = [], []
            for key in scenes.keys():
                idxs = scenes[key]
                cut = int(len(idxs)*0.95)
                train_set += idxs[:cut]
                val_set += idxs[cut:]
                #print(cut, len(train_set), len(val_set))

            if not eval_set:
                self.episodes_idx = train_set
            else:
                self.episodes_idx = val_set        
        
        self.number_of_episodes = len(self.episodes_idx)
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(options.root_path+'bert_model/')
        self.max_seq_length = 512 # maximum sequence length for instruction that BERT can take
        self.map_noise = options.map_noise


    def collect_stored_episodes(self, options, split):
        episodes_dir = options.stored_episodes_dir + split + "/"
        episodes_file_list = []
        _scenes_dir = os.listdir(episodes_dir)
        scenes_dir = [ x for x in _scenes_dir if os.path.isdir(episodes_dir+x) ]
        for scene in scenes_dir:
            for fil in os.listdir(episodes_dir+scene+"/"):
                to_add = False
                # add training examples only from the selected datasets
                for dataset in options.datasets:
                    if dataset in fil:
                        to_add = True
                if to_add:
                    episodes_file_list.append(episodes_dir+scene+"/"+fil)
        return episodes_file_list


    def __len__(self):
        return self.number_of_episodes


    def __getitem__(self, idx):
        #idx=0
        ind = self.episodes_idx[idx] # get the next episode index
        #ind = 0 # **** hardcode the ind for now for debugging
        ep_file = self.episodes_file_list[ind]
        #ep_file = "/home/ggeorgak/habitat-api/data/scene_datasets/mp3d_vln_episodes/val_unseen/2azQ1b91cZZ/ep_1_0_2azQ1b91cZZ_R2R_VLNCE_v1-2.npz"
        #ep_file = "/home/ggeorgak/habitat-api/data/scene_datasets/mp3d_vln_episodes_no_map/val_unseen/2azQ1b91cZZ/ep_1_0_2azQ1b91cZZ_R2R_VLNCE_v1-2.npz"
        #print(ep_file)
        #ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        #abs_pose = ep['abs_pose']
        goal_heatmap = torch.from_numpy(ep['goal_heatmap']) # num_poses x num_waypoints x 64 x 64
        map_semantic = torch.from_numpy(ep['map_semantic']) # num_poses x 1 x 192 x 192
        visible_waypoints = torch.from_numpy(ep['visible_waypoints']) # num_poses x num_waypoints
        covered_waypoints = torch.from_numpy(ep['covered_waypoints']) # num_poses x num_waypoints

        if self.vln_no_map or self.vln_L2M or self.vln_no_map_ext:
            step_ego_grid_maps = torch.from_numpy(ep['step_ego_grid_maps']) # num_poses x 3 x 192 x 192
            map_occupancy = torch.from_numpy(ep['map_occupancy']) # num_poses x 1 x 192 x 192
            ego_segm_maps = torch.from_numpy(ep['ego_segm_maps']) # num_poses x 27 x 192 x 192
            #print(step_ego_grid_maps.shape)
            #print(map_occupancy.shape)
            #print(ego_segm_maps.shape)


        #text_feat = torch.from_numpy(ep['text_feat'])
        # replicate text_feat for all number of poses in the episode
        #text_feat = text_feat.unsqueeze(0).repeat(map_semantic.shape[0], 1, 1)

        #goal_pose_coords = torch.tensor(ep['goal_rel_pose_coords'][0]).unsqueeze(0).unsqueeze(0)
        #viz_utils.write_tensor_imgSegm(img=map_semantic.cpu(), savepath="", name="gt_map_semantic"+str(0))
        #viz_utils.vis_heatmaps(goal_heatmap[0,:,:,:].squeeze(0), goal_heatmap[0,:,:,:].squeeze(0))
        #viz_utils.save_map_goal(map_semantic[0,:,:,:].unsqueeze(0).unsqueeze(0), torch.tensor([[[255,255]]]), goal_pose_coords, "", 0)

        # Should we use a single example from each episode at a time (to have more variety in each batch)?
        # For now stick to passing the entire sequence
        if self.sample_1: # forced to do this due to memory issues
            ind = random.randint(0,map_semantic.shape[0]-1) # Randomly choose one example from the sequence
            #ind=0
            goal_heatmap = goal_heatmap[ind,:,:,:].unsqueeze(0) # 1 x num_waypoints x h x w
            map_semantic = map_semantic[ind,:,:,:].unsqueeze(0)
            visible_waypoints = visible_waypoints[ind,:].unsqueeze(0)
            covered_waypoints = covered_waypoints[ind,:].unsqueeze(0)
            if self.vln_no_map or self.vln_L2M or self.vln_no_map_ext:
                step_ego_grid_maps = step_ego_grid_maps[ind,:,:,:].unsqueeze(0)
                map_occupancy = map_occupancy[ind,:,:,:].unsqueeze(0)
                ego_segm_maps = ego_segm_maps[ind,:,:,:].unsqueeze(0)

        #instruction = ep['instruction'].tostring().decode("utf-8")
        instruction = ep['instruction'].tobytes().decode("utf-8")
        instruction = "[CLS] " + instruction + " [SEP]"
        #print(instruction)
        
        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(instruction)
        #print(tokenized_text)
        # Map the token strings to their vocabulary indices.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        #print(tokens_tensor.shape)
        #print(segments_tensors.shape)

        # Truncate very large instructions to max length (512)
        if tokens_tensor.shape[1] > self.max_seq_length:
            tokens_tensor = tokens_tensor[:,:self.max_seq_length]
            segments_tensors = segments_tensors[:,:self.max_seq_length]

        ### Transform abs_pose to rel_pose
        #rel_pose = []
        #for i in range(abs_pose.shape[0]):
        #    rel_pose.append(utils.get_rel_pose(pos2=abs_pose[i,:], pos1=abs_pose[0,:]))

        item = {}

        if self.map_noise:
            map_semantic_noisy = utils.add_uniform_noise(tensor=map_semantic.clone(), a=-0.2, b=0.2)
            item['map_semantic'] = map_semantic_noisy
        else:
            item['map_semantic'] = map_semantic

        #item['pose'] = torch.from_numpy(np.asarray(rel_pose)).float()
        #item['abs_pose'] = torch.from_numpy(abs_pose).float()
        item['goal_heatmap'] = goal_heatmap # T x num_waypoints x 64 x 64
        item['map_semantic'] = map_semantic#_noisy # T x 1 x 192 x 192
        #item['map_semantic_noise_free'] = map_semantic
        item['visible_waypoints'] = visible_waypoints
        item['covered_waypoints'] = covered_waypoints
        #item['text_feat'] = text_feat
        item['tokens_tensor'] = tokens_tensor
        item['segments_tensors'] = segments_tensors

        if self.offline_eval:
            item['tokens'] = tokenized_text
            item['instruction'] = ep['instruction'] #instruction

        if self.vln_no_map or self.vln_L2M or self.vln_no_map_ext:
            item['step_ego_grid_maps'] = step_ego_grid_maps
            item['map_occupancy'] = map_occupancy
            item['ego_segm_maps'] = ego_segm_maps

        return item



### Dataloader for storing data in the unknown map case

class HabitatDataVLN_UnknownMap(Dataset):

    # Loads necessary data for the actual VLN task

    def __init__(self, options, config_file, scene_id, existing_episode_list=[], random_poses=False, pose_noise=1):

        self.options = options
        self.scene_id = scene_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_poses_per_example = options.num_poses_per_example

        self.parse_episodes(self.options.datasets)
        
        self.number_of_episodes = len(self.scene_data["episodes"])

        #print(self.number_of_episodes)
        #print(self.scene_data['episodes'][0])

        cfg = habitat.get_config(config_file)
        cfg.defrost()
        #cfg.SIMULATOR.SCENE = '/media/ggeorgak/DATA/habitat_backup/data/scene_datasets/mp3d/' + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        cfg.SIMULATOR.SCENE = options.root_path + options.scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        #cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        cfg.SIMULATOR.TURN_ANGLE = options.turn_angle
        cfg.SIMULATOR.FORWARD_STEP_SIZE = options.forward_step_size
        cfg.freeze()

        self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        seed = 0
        self.sim.seed(seed)

        self.hfov = float(cfg.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
        self.cfg_norm_depth = cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
        self.object_labels = options.n_object_classes
        self.spatial_labels = options.n_spatial_classes
        self.global_dim = (options.global_dim, options.global_dim)
        self.grid_dim = (options.grid_dim, options.grid_dim)
        self.cell_size = options.cell_size
        self.heatmap_size = (options.heatmap_size, options.heatmap_size)
        self.num_waypoints = options.num_waypoints
        self.min_angle_noise = np.radians(-15)
        self.max_angle_noise = np.radians(15)
        self.img_size = (options.img_size, options.img_size)
        self.img_segm_size = (options.img_segm_size, options.img_segm_size)
        self.normalize = True
        self.pixFormat = 'NCHW'
        #self.crop_size = (options.crop_size, options.crop_size)
        self.max_depth = cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.min_depth = cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH

        #self.preprocessed_scenes_dir = "/media/ggeorgak/DATA/habitat_backup/data/scene_datasets/mp3d_scene_pclouds/"
        self.preprocessed_scenes_dir = options.root_path + options.scenes_dir + "mp3d_scene_pclouds/"

         # get point cloud and labels of scene
        self.pcloud, self.label_seq_spatial, self.label_seq_objects = utils.load_scene_pcloud(self.preprocessed_scenes_dir,
                                                                                                    self.scene_id, self.object_labels)
        self.color_pcloud = utils.load_scene_color(self.preprocessed_scenes_dir, self.scene_id)

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
        K = np.array([
            [1 / np.tan(self.hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(self.hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
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


    def parse_episodes(self, sets):

        self.scene_data = {'episodes': []}

        for s in sets:

            if s=='R2R_VLNCE_v1-2':
                root_rxr_dir = self.options.root_path + "rxr-data/" + s + "/"
                episode_file = root_rxr_dir + self.options.split + "/" + self.options.split + ".json.gz"
                with gzip.open(episode_file, "rt") as fp:
                    self.data = json.load(fp)

                '''
                # count the number of episodes per scene
                scene_count={}
                dist_list = []
                for i in range(len(self.data['episodes'])):
                    sc_path = self.data['episodes'][i]['scene_id']
                    sc_id = sc_path.split('/')[-1].split('.')[0]
                    #start_pos = self.data['episodes'][i]['start_position']
                    #goal_pos = self.data['episodes'][i]['goals'][0]['position']
                    #if np.absolute(start_pos[1] - goal_pos[1]) >= 0.2:
                    #    continue
                    #ltg_dist = torch.linalg.norm(torch.tensor(start_pos)-torch.tensor(goal_pos))
                    #dist_list.append(ltg_dist.item())
                    if sc_id not in scene_count:
                        scene_count[sc_id] = 1
                    else:
                        scene_count[sc_id] += 1
                #print(scene_count)
                count_all=0
                sc_list = list(scene_count.keys())
                sc_list.sort()
                for sc in sc_list:
                    print(sc, scene_count[sc])
                    count_all+=scene_count[sc]
                #for sc in scene_count.keys():
                #    print(sc, scene_count[sc])
                #    count_all+=scene_count[sc]
                print(len(sc_list))
                print("All:", count_all)
                #print(dist_list)
                #print(np.mean(np.asarray(dist_list)))
                #with open('val_seen_dist.npy', 'wb') as f:
                #    np.save(f, dist_list)
                '''

                if self.options.split!="test":
                    # Load the gt information from R2R_VLNCE
                    episode_file_gt = self.options.root_path+"rxr-data/"+s+"_preprocessed/"+self.options.split +"/"+self.options.split+"_gt.json.gz"
                    with gzip.open(episode_file_gt, "rt") as fp:
                        self.data_gt = json.load(fp)
                
                #print(len(self.data['episodes']))
                #print(len(self.data_gt['episodes']))
                #print(len(self.data_gt.keys()))
                #print(self.data['episodes'][300])
                #print()
                #print(self.data_gt['episodes'][300])
                #self.instruction_vocab = self.data['instruction_vocab']
                # Need to keep only episodes that belong to current scene
                for i in range(len(self.data['episodes'])):
                    sc_path = self.data['episodes'][i]['scene_id']
                    sc_id = sc_path.split('/')[-1].split('.')[0]
                    if sc_id == self.scene_id:                        
                        # Check if given path has enough poses
                        #ref_path = self.data['episodes'][i]['reference_path']
                        #if len(ref_path) < self.num_poses_per_example:
                        #    continue
                        
                        # seems that the "start_rotation" for the R2R_VLNCE_v1-2 set has 180 degrees difference from the description
                        #self.data['episodes'][i]['start_rotation'] = utils.add_to_quarternion(rotation=self.data['episodes'][i]['start_rotation'], angle=-np.pi)
                        self.data['episodes'][i]['scene_id'] = self.scene_id
                        self.data['episodes'][i]['dataset'] = s
                        
                        if self.options.split!="test":
                            # check whether goal is at the same height as start position
                            start_pos = self.data['episodes'][i]['start_position']
                            goal_pos = self.data['episodes'][i]['goals'][0]['position']
                            if np.absolute(start_pos[1] - goal_pos[1]) >= 0.2 and self.options.check_floor:
                                continue
                            # get gt info
                            gt_info = self.data_gt[ str(self.data['episodes'][i]['episode_id']) ] # locations, forward_steps, actions
                            #print(len(gt_info['locations']), gt_info['forward_steps'], len(gt_info['actions']))
                            self.data['episodes'][i]['waypoints'] = gt_info['locations']
                            self.data['episodes'][i]['actions'] = gt_info['actions']

                        self.scene_data['episodes'].append(self.data['episodes'][i])
                            

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

            if self.cfg_norm_depth:
                depth = utils.unnormalize_depth(depth, min=self.min_depth, max=self.max_depth)

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