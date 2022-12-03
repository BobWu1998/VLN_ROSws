import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataloader import HabitatDataVLN_UnknownMap
# #from models.predictors import get_predictor_from_options
# #from models.img_segmentation import get_img_segmentor_from_options
from models import get_model_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import test_utils as tutils
from models.semantic_grid import SemanticGrid
import os
import matplotlib.pyplot as plt
import seaborn as sns

import random
from planning.ddppo_policy import DdppoPolicy
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
from realsense_read import record_rgbd
from move_new_v2 import robot

class VLNTesterUnknownMap(object):
    def __init__(self, options): # (self, options, scene_id):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # build summary dir
        summary_dir = os.path.join(self.options.log_dir)#, scene_id)
        summary_dir = os.path.join(summary_dir, 'tensorboard')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        # tensorboardX SummaryWriter for use in save_summaries
        self.summary_writer = SummaryWriter(summary_dir)

        self.test_ds = HabitatDataVLN_UnknownMap(self.options) #, config_file=self.options.config_test_file), scene_id=self.scene_id)

        # Load the goal predictor model
        self.goal_pred_model = get_model_from_options(self.options)
        # if torch.cuda.is_available():
        #     self.goal_pred_model.cuda()
        self.models_dict = {'goal_pred_model':self.goal_pred_model}

        print("Using ", torch.cuda.device_count(), "gpus")
        for k in self.models_dict:
            self.models_dict[k] = nn.DataParallel(self.models_dict[k])

        latest_checkpoint = tutils.get_latest_model(save_dir=self.options.model_exp_dir)
        self.models_dict = tutils.load_model(models=self.models_dict, checkpoint_file=latest_checkpoint)
        print("Model loaded checkpoint:", latest_checkpoint)
        self.models_dict["goal_pred_model"].eval()

        # init local policy model
        if self.options.local_policy_model=="4plus":
            model_ext = 'gibson-4plus-mp3d-train-val-test-resnet50.pth'
        else:
            model_ext = 'gibson-2plus-resnet50.pth'
        model_path = self.options.root_path + "local_policy_models/" + model_ext
        self.l_policy = DdppoPolicy(path=model_path)
        self.l_policy = self.l_policy.to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length = 512 # maximum sequence length for instruction that BERT can take

    def pipeline_integration(self):

        with torch.no_grad():
            robot_ins = robot()
            t = 0
            idx = 0 # redundant since only one test set (real world)
            abs_poses = []
            agent_height = []
            sim_agent_poses = [] # for estimating the metrics at the end of the episode
            ltg_counter=0
            ltg_abs_coords = torch.zeros((1, 1, 2), dtype=torch.int64).to(self.device)
            ltg_abs_coords_list = []
            
            x, y, z, angle_x, angle_y, angle_z = robot_ins.get_pose()
            previous_pos = [x, y, angle_z]
            y_height = 1.25 
            while t < 500: # max timestep
                x, y, z, angle_x, angle_y, angle_z = robot_ins.get_pose()
                agent_pose = x, y, angle_z
                abs_poses.append([x, y, angle_z])
                agent_height.append(y_height)

                # # getting rgbd input
                # img_name = '/home/bo/VLN_realsense/data/realsense/rgb.png' # this is the rgb image name loaded from record_rgbd(), don't delete
                # depth_filename = '/home/bo/Documents/0809_6th/_Depth_1660076119478.12084960937500.csv'
                rgb_arr, depth_arr = record_rgbd()


                sg = SemanticGrid(1, self.test_ds.grid_dim, self.options.heatmap_size, self.options.cell_size,
                                            spatial_labels=self.options.n_spatial_classes, object_labels=self.options.n_object_classes)
                sg_global = SemanticGrid(1, self.test_ds.global_dim, self.options.heatmap_size, self.options.cell_size,
                    spatial_labels=self.options.n_spatial_classes, object_labels=self.options.n_object_classes)                       

                ### Ground Projecting Depth Value
                depth_loaded = depth_arr

                imageSize = (self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
                # crop depth_abs to img_size
                depth_abs = torch.tensor(depth_loaded, device='cuda')
                
                # previous xs and ys
                xs, ys = torch.tensor(np.meshgrid(np.linspace(-1,1, depth_abs.shape[1]), np.linspace(1,-1,depth_abs.shape[0])), device='cuda')
                xs = xs.reshape(1,depth_abs.shape[0],depth_abs.shape[1])
                ys = ys.reshape(1,depth_abs.shape[0],depth_abs.shape[1])
                
                # local3D_step = utils.depth_to_3D(depth_abs, self.test_ds.img_size, self.test_ds.xs, self.test_ds.ys, self.test_ds.inv_K)
                local3D_step = utils.depth_to_3D(depth_abs, self.test_ds.img_size, xs, ys, self.test_ds.inv_K)
                
                # viz_utils.vis_arr(local3D_step.reshape(depth_abs.shape[0], depth_abs.shape[1], 3)[:,:,0], './', 'x') 
                # viz_utils.vis_arr(local3D_step.reshape(depth_abs.shape[0], depth_abs.shape[1], 3)[:,:,1], './', 'height') 
                
                # [1, 3, 512, 512]
                ego_grid_sseg_3 = map_utils.est_occ_from_depth([local3D_step], grid_dim=self.test_ds.global_dim, cell_size=self.test_ds.cell_size, 
                                                                        device=self.device, occupancy_height_thresh=self.options.occupancy_height_thresh)    

                ### Semantic Segmentation with deeplabv3
                # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
                # or any of these variants
                model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
                # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
                model.eval()
                # sample execution (requires torchvision)
                
                # input_image = Image.open(img_name)
                input_image = Image.fromarray(rgb_arr)
                input_image = input_image.convert("RGB")
                preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                # prepare RGB image
                img_tensor = preprocess(input_image)

                # prepare depth image
                depth_abs = torch.tensor(depth_loaded.reshape(imageSize), device='cuda')
                depth_img = depth_abs.clone().permute(2,0,1).unsqueeze(0)
                depth_img = F.interpolate(depth_img, size=self.test_ds.img_segm_size, mode='nearest')
                
                # put everything in the batch
                segm_batch = {'images':img_tensor.unsqueeze(0).unsqueeze(0),
                    'depth_imgs':depth_img.to(self.device).unsqueeze(0)}

                pred_ego_crops_sseg, img_segm = utils.run_img_segm(model=model, 
                                                        input_batch=segm_batch, 
                                                        object_labels=self.test_ds.object_labels, 
                                                        crop_size=self.test_ds.grid_dim, 
                                                        cell_size=self.test_ds.cell_size,
                                                        xs=self.test_ds._xs,
                                                        ys=self.test_ds._ys,
                                                        inv_K=self.test_ds.inv_K,
                                                        points2D_step=self.test_ds._points2D_step)

                pred_ego_crops_sseg = map_utils.update_sseg_with_occ(pred_ego_crops_sseg, ego_grid_sseg_3, self.test_ds.grid_dim)


                # Pseudo Poses
                rel_abs_pose = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
                _rel_abs_pose = torch.Tensor(rel_abs_pose).unsqueeze(0).float()       
                _rel_abs_pose = _rel_abs_pose.to(self.device)
                abs_pose_coords = tutils.get_coord_pose(sg_global, _rel_abs_pose, abs_poses[0], self.test_ds.global_dim[0], self.test_ds.cell_size, self.device) # B x T x 3

                # We operate in egocentric coords so agent should always be in the middle of the map
                rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[t])
                _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
                _rel_pose = _rel_pose.to(self.device)
                pose_coords = tutils.get_coord_pose(sg, _rel_pose, abs_poses[t], self.test_ds.grid_dim[0], self.test_ds.cell_size, self.device) # B x T x 3
                
                # need to keep track of the starting point in the original view
                rel_pose_0 = utils.get_rel_pose(pos2=abs_poses[0], pos1=abs_poses[t])
                _rel_pose_0 = torch.Tensor(rel_pose_0).unsqueeze(0).float()
                _rel_pose_0 = _rel_pose_0.to(self.device)
                pose_coords_0 = tutils.get_coord_pose(sg, _rel_pose_0, abs_poses[t], self.test_ds.grid_dim[0], self.test_ds.cell_size, self.device)

                gt_waypoints_pose_coords = pose_coords_0[0]

                step_ego_grid_maps = map_utils.crop_grid(grid=ego_grid_sseg_3, crop_size=self.test_ds.grid_dim)
                step_ego_grid_maps = step_ego_grid_maps.unsqueeze(0)
                step_ego_grid_maps = step_ego_grid_maps.to('cpu')
                pred_ego_crops_sseg = pred_ego_crops_sseg.to('cpu')

                # run goal prediction
                # instruction = 'Go straight all the way down to pass the chairs, once pass those chairs make a right turn and go straight.' ## this instruction gives a good output
                # instruction = 'Go straight to pass the chairs, once pass those chairs make a right turn and go straight'
                # instruction = 'Go straight to pass the chair, once pass the chair make a right turn and go straight'
                # instruction = 'Go straight all the way down to pass the chairs, once pass those chairs make a left turn and go straight.' 
                # instruction = 'Go straight all the way down to pass the chairs and the tables, once pass those chairs make a left turn and go straight.'
                # instruction = 'Go straight all the way down to past the chair, once past the chair make a right turn and go straight.'
                # instruction = 'Walk forward to the chair, once past the chair, turn right and walk down the hallway.'
                instruction = 'Walk towards the chair, once past the chair, turn right and walk down the hallway.'
                
                
                start_pose = [0.0, 0.0, 0.0] # the start_pose should always be the origin - [0, 0, 0]

                # step_ego_grid_maps is the one dominating the problem
                # start position is supposed to be in simulator
                pred_waypoints_pose_coords, pred_waypoints_vals, waypoints_cov, pred_maps_spatial, pred_maps_objects = self.run_goal_pred(instruction, sg=sg, ego_occ_maps=step_ego_grid_maps, 
                                                                                                                                    ego_segm_maps=pred_ego_crops_sseg, start_pos=start_pose, pose=abs_poses[t])
            
                # np.save('/home/bo/Desktop/pred_maps_spatial_robot.npy', pred_maps_spatial.to('cpu').detach().numpy())

                
                
                # Option to save visualizations of steps
                pred_waypoints_pose_coords = torch.cat( (gt_waypoints_pose_coords[0,:].unsqueeze(0), pred_waypoints_pose_coords.squeeze(0)), dim=0 ) # 10 x 2
                
                # And assume the initial waypoint is covered
                waypoints_cov = torch.cat( (torch.tensor([1]).to(self.device), waypoints_cov.squeeze(0)), dim=0 ) # 10

                ltg_dist = torch.linalg.norm(ltg_abs_coords.clone().float().cpu()-abs_pose_coords.float().cpu())*self.options.cell_size # distance to current long-term goal


                # Estimate long term goal
                if ((ltg_counter % self.options.steps_after_plan == 0) or  # either every k steps
                    (ltg_dist < 0.2)): # or we reached ltg

                    goal_confidence = pred_waypoints_vals[0,-1]

                    # if waypoint goal confidence is low then remove it from the waypoints list
                    if goal_confidence < self.options.goal_conf_thresh:
                        pred_waypoints_pose_coords[-1,0], pred_waypoints_pose_coords[-1,1] = -200, -200 

                    # Choose the waypoint following the one that is closest to the current location
                    pred_waypoint_dist = np.linalg.norm(pred_waypoints_pose_coords.cpu().numpy() - pose_coords.squeeze(0).cpu().numpy(), axis=-1)
                    min_point_ind = torch.argmin(torch.tensor(pred_waypoint_dist))
                    if min_point_ind >= pred_waypoints_pose_coords.shape[0]-1:
                        min_point_ind = pred_waypoints_pose_coords.shape[0]-2
                    if pred_waypoints_pose_coords[min_point_ind+1][0]==-200: # case when min_point_ind+1 is goal waypoint but it has low confidence
                        min_point_ind = min_point_ind-1
                    ltg = pred_waypoints_pose_coords[min_point_ind+1].unsqueeze(0).unsqueeze(0)

                    # To keep the same goal in multiple steps first transform ego ltg to abs global coords 
                    ltg_abs_coords = tutils.transform_ego_to_geo(ltg, pose_coords, abs_pose_coords, abs_poses, t)
                    ltg_abs_coords_list.append(ltg_abs_coords)

                    ltg_counter = 0 # reset the ltg counter
                ltg_counter += 1


                # transform ltg_abs_coords to current egocentric frame for visualization
                ltg_sim_abs_pose = tutils.get_3d_pose(pose_2D=ltg_abs_coords.clone().squeeze(0), agent_pose_2D=abs_pose_coords.clone().squeeze(0), agent_sim_pose=agent_pose[:2], 
                                                            y_height=y_height, init_rot=torch.tensor(abs_poses[0][2]), cell_size=self.options.cell_size)
                ltg_ego_coords, _ = tutils.transform_to_map_coords(sg=sg, position=ltg_sim_abs_pose, abs_pose=abs_poses[t], 
                                                                            grid_size=self.test_ds.grid_dim[0], cell_size=self.options.cell_size, device=self.device)

                if self.options.save_nav_images:
                    # save_img_dir_ = self.options.log_dir + "/" + self.scene_id + "/" + self.options.save_img_dir+'ep_'+str(idx)+'/'
                    save_img_dir_ = self.options.log_dir + "/" + 'real_world' + "/" + self.options.save_img_dir+'ep_'+str(idx)+'/'
                    if not os.path.exists(save_img_dir_):
                        os.makedirs(save_img_dir_)

                    print('save_img_dir_', save_img_dir_)
                    ### saves egocentric rgb, depth observations
                    # img = input_image
                    # viz_utils.display_sample(img.cpu().numpy(), np.squeeze(depth_abs.cpu().numpy()), t, savepath=save_img_dir_)
                    input_image.save(save_img_dir_+str(t)+"rgb_input.png")

                    # add heatmap
                    sns.heatmap(depth_arr)
                    # save the figure
                    plt.savefig(save_img_dir_+str(t)+"depth_input.png")

                    ### visualize the predicted waypoints vs gt waypoints in gt semantic egocentric frame
                    viz_utils.show_waypoint_pred(pred_maps_objects.squeeze(0).squeeze(0), ltg=ltg_ego_coords.clone().cpu().numpy(), pose_coords=pose_coords.clone().cpu().numpy(), num_points=self.options.num_waypoints,
                                                        pred_waypoints=pred_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_pred.png")
                    # viz_utils.show_waypoint_pred(pred_maps_objects.squeeze(0).squeeze(0), ltg=None, pose_coords=None, num_points=self.options.num_waypoints,
                    #                         pred_waypoints=pred_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_pred.png")

                    # # viz_utils.show_waypoint_pred(gt_map_semantic, num_points=self.options.num_waypoints,
                    # #                                     gt_waypoints=gt_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_gt.png")

                    # ### visualize the episode steps in the global geocentric frame
                    # gt_map_semantic_global, _ = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[0],
                    #                                         grid_dim=self.test_ds.global_dim, cell_size=self.options.cell_size, color_pcloud=color_pcloud, z=z)
                    # # project gt waypoints in global geocentric frame
                    # gt_waypoints_global = torch.zeros((len(gt_waypoints), 2))
                    # for k in range(len(gt_waypoints)):
                    #     point_global_coords, _ = tutils.transform_to_map_coords(sg=sg_global, position=gt_waypoints[k], abs_pose=abs_poses[0], 
                    #                                                             grid_size=self.test_ds.global_dim[0], cell_size=self.options.cell_size, device=self.device)
                    #     gt_waypoints_global[k,:] = point_global_coords.squeeze(0).squeeze(0)                                  
                    # # transform predicted waypoints in global geocentric frame
                    # pred_waypoints_global = torch.zeros((len(pred_waypoints_pose_coords), 2))
                    # for k in range(len(pred_waypoints_pose_coords)):
                    #     pred_point_global_coords = tutils.transform_ego_to_geo(pred_waypoints_pose_coords[k].unsqueeze(0).unsqueeze(0), pose_coords, abs_pose_coords, abs_poses, t)
                    #     pred_waypoints_global[k,:] = pred_point_global_coords.squeeze(0)

                    # # viz_utils.show_waypoint_pred(gt_map_semantic_global, ltg=ltg_abs_coords.clone().cpu().numpy(), pose_coords=abs_pose_coords.clone().cpu().numpy(), num_points=self.options.num_waypoints,
                    # #                                     pred_waypoints=pred_waypoints_global, gt_waypoints=gt_waypoints_global, savepath=save_img_dir_+"global_"+str(t)+"_waypoints.png")

                    # saves predicted areas (egocentric)
                    # viz_utils.vis_arr(pred_ego_crops_sseg)
                    viz_utils.save_map_pred_steps(step_ego_grid_maps, pred_maps_spatial, 
                                                        pred_maps_objects, pred_ego_crops_sseg, save_img_dir_, t, instruction)

                    viz_utils.write_tensor_imgSegm(img_segm, save_img_dir_, name="img_segm", t=t, labels=21)
                ##### Action decision process #####
                pred_goal = pred_waypoints_pose_coords[-1].unsqueeze(0).unsqueeze(0)
                pred_goal_dist = torch.linalg.norm(pred_goal.clone().float()-pose_coords.float())*self.options.cell_size # distance to current predicted goal

                depth = depth_abs.type(torch.FloatTensor).to(self.device)
                action_id = self.run_local_policy(depth=depth, goal=ltg_abs_coords.clone(),
                                                            pose_coords=abs_pose_coords.clone(), rel_agent_o=rel_abs_pose[2], step=t)
                print('action_id:', action_id)
                # if stop is selected from local policy then randomly select an action
                if action_id==0:
                    print('stop action id, randomly choose one action')
                    action_id = random.randint(1,3)
                # Let's apply the action
                if action_id==1:
                    print('forward')
                    # robot_ins.get_pose()
                    robot_ins.forward()
                    # robot_ins.get_pose()
                    # move_around.forward()
                elif action_id==2:
                    print('left')
                    # robot_ins.get_pose()
                    robot_ins.rotate(clockwise=0)
                    # robot_ins.get_pose()
                    # move_around.forward()
                elif action_id==3:
                    print('right')
                    # robot_ins.get_pose()
                    robot_ins.rotate(clockwise=1)
                    # robot_ins.get_pose()
                    # move_around.forward()
                robot_ins.get_pose()


                print('---------------------------------------------------------------------------')
                print('time step:', t)
                print('agent_pose:', agent_pose) # abs pose from simulator # abs_poses.append(agent_pose)
                print('y_height:', y_height)
                print('rel_abs_pose:', rel_abs_pose)
                print('abs_pose_coords', abs_pose_coords)
                print('pose_coords', pose_coords)
                print('action_id:', action_id)
                print('ltg_abs_coords:', ltg_abs_coords)
                print('pose_coords_0', pose_coords_0)
                print('ltg_ego_coords', ltg_ego_coords)
                print('---------------------------------------------------------------------------')
                t += 1
                
    def run_goal_pred(self, instruction, sg, ego_occ_maps, ego_segm_maps, start_pos, pose):
        ## Prepare model inputs
        instruction = "[CLS] " + instruction + " [SEP]"
        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(instruction)
        # Map the token strings to their vocabulary indices.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Truncate very large instructions to max length (512)
        if tokens_tensor.shape[1] > self.max_seq_length:
            tokens_tensor = tokens_tensor[:,:self.max_seq_length]
            segments_tensors = segments_tensors[:,:self.max_seq_length]

        # Get the heatmap for the start position (initial waypoint)
        # This is with respect to the agent's current location
        point_pose_coords, visible = tutils.transform_to_map_coords(sg=sg, position=start_pos, abs_pose=pose, 
                                                                        grid_size=self.test_ds.grid_dim[0], cell_size=self.options.cell_size, device=self.device)
        start_pos_heatmap = utils.locs_to_heatmaps(keypoints=point_pose_coords.squeeze(0), img_size=self.test_ds.grid_dim, out_size=self.test_ds.heatmap_size)

        input_batch = {'step_ego_grid_maps':ego_occ_maps,
                       'ego_segm_maps':ego_segm_maps,
                       'goal_heatmap':start_pos_heatmap.unsqueeze(0).unsqueeze(0), # using the default name in the vln_goal_predictor
                       'tokens_tensor':tokens_tensor.unsqueeze(0),
                       'segments_tensors':segments_tensors.unsqueeze(0)}

        pred_output = self.models_dict['goal_pred_model'](input_batch)

        pred_waypoints_heatmaps = pred_output['pred_waypoints']
        waypoints_cov_prob = pred_output['waypoints_cov']

        # Convert predicted heatmaps to map coordinates
        pred_waypoints_resized = F.interpolate(pred_waypoints_heatmaps, size=(self.test_ds.grid_dim[0], self.test_ds.grid_dim[1]), mode='nearest')
        pred_locs, pred_vals = utils.heatmaps_to_locs(pred_waypoints_resized)
        # get the predicted coverage
        waypoints_cov = torch.argmax(waypoints_cov_prob, dim=2)

        pred_maps_objects = pred_output['pred_maps_objects']
        pred_maps_spatial = pred_output['pred_maps_spatial']

        return pred_locs, pred_vals, waypoints_cov, pred_maps_spatial, pred_maps_objects


    def run_local_policy(self, depth, goal, pose_coords, rel_agent_o, step):
        planning_goal = goal.squeeze(0).squeeze(0)
        planning_pose = pose_coords.squeeze(0).squeeze(0)
        sq = torch.square(planning_goal[0]-planning_pose[0])+torch.square(planning_goal[1]-planning_pose[1])
        rho = torch.sqrt(sq.float())
        phi = torch.atan2((planning_pose[0]-planning_goal[0]),(planning_pose[1]-planning_goal[1]))
        phi = phi - rel_agent_o
        rho = rho*self.test_ds.cell_size
        point_goal_with_gps_compass = torch.tensor([rho,phi], dtype=torch.float32).to(self.device)
        depth = depth.reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
        return self.l_policy.plan(depth, point_goal_with_gps_compass, step)


