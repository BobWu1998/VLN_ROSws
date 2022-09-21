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
# from sklearn.metrics import confusion_matrix
# import metrics
# import json
# import cv2
import random
# import math
# import imageio
from planning.ddppo_policy import DdppoPolicy
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
import move_around
from realsense_read import record_rgbd
from move_new_v2 import robot
# import move_around

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


    def occ_proj(self):
        file = open('/home/bo/Desktop/VLN_desktop/VLN_realworld/frames/real_data/_Depth_1655240137885.53833007812500.csv', 'rb')
        depth_abs = np.loadtxt(file,delimiter = ",")
        img = imageio.imread('/home/bo/Desktop/VLN_desktop/VLN_realworld/frames/real_data/_Color_1655240137885.55224609375000.png')
        print('log-----')

        imageSize = (self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
        # crop depth_abs to img_size
        ###
        # crop the depth_abs (not actually needed)
        # depth_abs = depth[depth.shape[0]//2-self.test_ds.img_size[0]//2:depth.shape[0]//2+self.test_ds.img_size[0]//2, depth.shape[1]//2-self.test_ds.img_size[1]//2:depth.shape[1]//2+self.test_ds.img_size[1]//2].reshape(imageSize)
        # depth_abs = depth_abs[0:self.test_ds.img_size[0], 300:1020].reshape(imageSize)#-self.test_ds.img_size[1]:].reshape(imageSize)#, 0:self.test_ds.img_size[1]].reshape(imageSize)#, 300:1020].reshape(imageSize)#-self.test_ds.img_size[1]:].reshape(imageSize)
        depth_abs = torch.tensor(depth_abs, device='cuda')
        print("depth_abs size cuda ",depth_abs.size())
        
        xs, ys = torch.tensor(np.meshgrid(np.linspace(-1,1, depth_abs.shape[1]), np.linspace(1,-1,depth_abs.shape[0])), device='cuda')
        xs = xs.reshape(1,depth_abs.shape[0],depth_abs.shape[1])
        ys = ys.reshape(1,depth_abs.shape[0],depth_abs.shape[1])
        # local3D_step = utils.depth_to_3D(depth_abs, self.test_ds.img_size, self.test_ds.xs, self.test_ds.ys, self.test_ds.inv_K)
        local3D_step = utils.depth_to_3D(depth_abs, self.test_ds.img_size, xs, ys, self.test_ds.inv_K)
        
        viz_utils.vis_arr(local3D_step.reshape(depth_abs.shape[0], depth_abs.shape[1], 3)[:,:,0], './', 'x') 
        viz_utils.vis_arr(local3D_step.reshape(depth_abs.shape[0], depth_abs.shape[1], 3)[:,:,1], './', 'height') 
        
        # viz_utils.vis_arr(local3D_step.reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 3)[:,:,1], './', 'height') 
        # viz_utils.vis_arr(local3D_step.reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 3)[:,:,2], './', 'depth') # visualize z
        # print("local3D_step", local3D_step.shape)
        # plt.imshow(img[depth.shape[0]//2-self.test_ds.img_size[0]//2:depth.shape[0]//2+self.test_ds.img_size[0]//2, depth.shape[1]//2-self.test_ds.img_size[1]//2:depth.shape[1]//2+self.test_ds.img_size[1]//2])
        
        # plt.savefig("./cropped_depth.png")
        ego_grid_sseg_3 = map_utils.est_occ_from_depth([local3D_step], grid_dim=self.test_ds.global_dim, cell_size=self.test_ds.cell_size, 
                                                                device=self.device, occupancy_height_thresh=self.options.occupancy_height_thresh)
        print("ego_grid_sseg_3.size", ego_grid_sseg_3.size())
        viz_utils.write_img(ego_grid_sseg_3, './', 'depth_test')

        print("ego_grid_sseg_3", ego_grid_sseg_3.shape)
    
    def img_seg(self):
        # read the depth and img file


        depth_imgs = {
            0: '_Depth_1655240137885.53833007812500.csv',
            1: '_Depth_1655325311485.18041992187500.csv',
            2: '_Depth_1655325321484.36279296875000.csv',
            3: '_Depth_1655325328083.60302734375000.csv',
        }

        imgs = {
            0: '_Color_1655240137885.55224609375000.png',
            1: '_Color_1655325311485.19458007812500.png',
            2: '_Color_1655325321484.37670898437500.png',
            3: '_Color_1655325328083.61694335937500.png',
        }


        for img_idx in range(len(imgs)):
            file = open(self.options.root_path + 'frames/' + depth_imgs[img_idx], 'rb')
            depth_abs = np.loadtxt(file,delimiter = ",")
            img = imageio.imread(self.options.root_path + 'frames/' + imgs[img_idx])
            print('log-----')
            # viz_utils.display_sample(img, depth_abs, 0, savepath='./')
            imageSize = (self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
            # crop depth_abs to img_size, send to cuda and interpolate
            depth_abs = depth_abs[0:self.test_ds.img_size[0], 0:self.test_ds.img_size[1]]
            depth_abs = torch.tensor(depth_abs.reshape(imageSize), device='cuda')
            depth_abs = depth_abs.clone().permute(2,0,1).unsqueeze(0)
            depth_img = F.interpolate(depth_abs, size=self.test_ds.img_segm_size, mode='nearest')
            
            # crop img and send to cuda
            # imgData = utils.preprocess_img(img, cropSize=self.test_ds.img_segm_size, pixFormat=self.test_ds.pixFormat, normalize=self.test_ds.normalize)
            # img = img[0:self.test_ds.img_size[0], 500:1012]#0:self.test_ds.img_size[1]]
            img = img[-self.test_ds.img_size[0]:, 400:912]#0:self.test_ds.img_size[1]]
            
        
            plt.imshow(img)
            plt.savefig(self.options.root_path + 'segm_test/img'+ str(img_idx))
            # plt.show()
            img = torch.tensor(img, device='cuda')

            imgData = utils.preprocess_img(img, cropSize=self.test_ds.img_size[0], pixFormat=self.test_ds.pixFormat, normalize=self.test_ds.normalize)
            
            segm_batch = {'images':imgData.to(self.device).unsqueeze(0).unsqueeze(0),
                                    'depth_imgs':depth_img.to(self.device).unsqueeze(0)}

            pred_img_segm = self.test_ds.img_segmentor(segm_batch)
            img_segm = pred_img_segm['pred_segm'].squeeze(0)
            img_labels = torch.argmax(pred_img_segm['pred_segm'].detach(), dim=2, keepdim=True) # B x T x 1 x cH x cW
            viz_utils.write_tensor_imgSegm(img_segm, self.options.root_path + '/segm_test/', name='img_seg'+ str(img_idx), t=0, labels=27)

        # viz_utils.display_sample(img, depth_abs, 0)
        

    def img_seg_habitat(self):
        def rgba2rgb( rgba, background=(255,255,255) ):
            row, col, ch = rgba.shape

            if ch == 3:
                return rgba

            assert ch == 4, 'RGBA image has 4 channels.'

            rgb = np.zeros( (row, col, 3), dtype='float32' )
            r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

            a = np.asarray( a, dtype='float32' ) / 255.0

            R, G, B = background

            rgb[:,:,0] = r * a + (1.0 - a) * R
            rgb[:,:,1] = g * a + (1.0 - a) * G
            rgb[:,:,2] = b * a + (1.0 - a) * B

            return np.asarray( rgb, dtype='uint8' )
        # read the depth and img file


        depth_imgs = {
            0: '_Depth_1655240137885.53833007812500.csv',
            1: '_Depth_1655240137885.53833007812500.csv',
            2: '_Depth_1655240137885.53833007812500.csv',
            3: '_Depth_1655240137885.53833007812500.csv',
            4: '_Depth_1655240137885.53833007812500.csv',
            5: '_Depth_1655240137885.53833007812500.csv',
            6: '_Depth_1655240137885.53833007812500.csv',
            7: '_Depth_1655240137885.53833007812500.csv',
            8: '_Depth_1655240137885.53833007812500.csv',
        }

        imgs = {
            0: 'rgba1.png',
            1: 'rgba2.png',
            2: 'rgba3.png',
            3: 'rgba4.png',
            4: 'rgba5.png',
            5: 'rgba6.png',
            6: 'rgba7.png',
            7: 'rgba8.png',
            8: 'rgba9.png',
        }

        scene_id = "1pXnuDYAj8r"#"2azQ1b91cZZ"
        for img_idx in range(len(imgs)):
            file = open(self.options.root_path + 'frames/' + scene_id +'_img/' + depth_imgs[img_idx], 'rb')
            depth_abs = np.loadtxt(file,delimiter = ",")
            img = imageio.imread(self.options.root_path + 'frames/' + scene_id +'_img/' + imgs[img_idx])
            print('log-----')
            # viz_utils.display_sample(img, depth_abs, 0, savepath='./')
            imageSize = (self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
            # crop depth_abs to img_size, send to cuda and interpolate
            depth_abs = depth_abs[0:self.test_ds.img_size[0], 0:self.test_ds.img_size[1]]
            depth_abs = torch.tensor(depth_abs.reshape(imageSize), device='cuda')
            depth_abs = depth_abs.clone().permute(2,0,1).unsqueeze(0)
            depth_img = F.interpolate(depth_abs, size=self.test_ds.img_segm_size, mode='nearest')
            
            # crop img and send to cuda
            # imgData = utils.preprocess_img(img, cropSize=self.test_ds.img_segm_size, pixFormat=self.test_ds.pixFormat, normalize=self.test_ds.normalize)
            # img = img[0:self.test_ds.img_size[0], 500:1012]#0:self.test_ds.img_size[1]]
            #img = img[-self.test_ds.img_size[0]:, 400:912]#0:self.test_ds.img_size[1]]
            
            img = rgba2rgb(img)[-self.test_ds.img_size[0]:, :self.test_ds.img_size[1]]#0:self.test_ds.img_size[1]]
            
        
            plt.imshow(img)
            plt.savefig(self.options.root_path + 'segm_test/' + scene_id + '_img'+ str(img_idx))
            # plt.show()
            img = torch.tensor(img, device='cuda')

            imgData = utils.preprocess_img(img, cropSize=self.test_ds.img_size[0], pixFormat=self.test_ds.pixFormat, normalize=self.test_ds.normalize)
            # print(imgData.size(), depth_img.size())
            segm_batch = {'images':imgData.to(self.device).unsqueeze(0).unsqueeze(0),
                                    'depth_imgs':depth_img.to(self.device).unsqueeze(0)}

            # ### Pure Semantic Segmentation
            # pred_img_segm = self.test_ds.img_segmentor(segm_batch)
            # img_segm = pred_img_segm['pred_segm'].squeeze(0)
            # img_labels = torch.argmax(pred_img_segm['pred_segm'].detach(), dim=2, keepdim=True) # B x T x 1 x cH x cW
            # viz_utils.write_tensor_imgSegm(img_segm, self.options.root_path + '/segm_test/', name=scene_id + '_img'+ str(img_idx), t=0, labels=27)


            pred_ego_crops_sseg, img_segm = utils.run_img_segm(model=self.test_ds.img_segmentor, 
                                        input_batch=segm_batch, 
                                        object_labels=self.test_ds.object_labels, 
                                        crop_size=self.test_ds.grid_dim, 
                                        cell_size=self.test_ds.cell_size,
                                        xs=self.test_ds._xs,
                                        ys=self.test_ds._ys,
                                        inv_K=self.test_ds.inv_K,
                                        points2D_step=self.test_ds._points2D_step)
            # plt.figure(figsize=(12 ,8))
            # plt.axis('off')
            # plt.imshow(pred_ego_crops_sseg.to('cpu'))
            # plt.show()
            # plt.savefig(save_img_dir_+str(t)+"_im_ego_img_segm.png", bbox_inches='tight', pad_inches=0, dpi=50) # 100
            # plt.close()    
            
# viz_utils.display_sample(img, depth_abs, 0)

    def seg_proj(self):
        file = open('/home/bo/Desktop/VLN_desktop/VLN_realworld/frames/real_data/_Depth_1655240137885.53833007812500.csv', 'rb')
        depth_abs = np.loadtxt(file,delimiter = ",")
        
        img = imageio.imread('/home/bo/Desktop/VLN_desktop/VLN_realworld/frames/real_data/_Color_1655240137885.55224609375000.png')        

        ## prediction with deeplabv3

        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # or any of these variants
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
        model.eval()
        # sample execution (requires torchvision)
        from PIL import Image
        from torchvision import transforms
        img_name = '/home/bo/Desktop/VLN_desktop/VLN_realworld/frames/real_data/_Color_1655240137885.55224609375000.png'
        input_image = Image.open(img_name)
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # prepare RGB image
        img_tensor = preprocess(input_image)

        # prepare depth image
        imageSize = (self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
        depth_abs = torch.tensor(depth_abs.reshape(imageSize), device='cuda')
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
        print('pred_ego_crops_sseg', pred_ego_crops_sseg.size())
        color_ego_img_segm = viz_utils.colorize_grid(pred_ego_crops_sseg, color_mapping=27)
        im_ego_img_segm = color_ego_img_segm[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        plt.imshow(im_ego_img_segm)
        plt.show()


    def run_goal_pred(self, instruction, sg, ego_occ_maps, ego_segm_maps, start_pos, pose):
        # print('Instruction:', instruction)
        # # print('sg.size()',sg.size())
        # print('ego_occ_maps, ego_segm_maps', ego_occ_maps.size(), ego_segm_maps.size())
        # print('start_pos, pose', start_pos, pose)
        # print(ego_occ_maps.get_device())

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

    def update_sseg(self):
        # # # data from 6th floor Levine
        # depth_filename = '/home/bo/Desktop/VLN_desktop/VLN_realworld/frames/real_data/_Depth_1655240137885.53833007812500.csv'
        # img_name = '/home/bo/Desktop/VLN_desktop/VLN_realworld/frames/real_data/_Color_1655240137885.55224609375000.png'

        # # data from GRASP Lab
        # depth_filename = '/home/bo/Documents/160945/_Depth_1659125385101.41918945312500.csv' 
        # img_name = '/home/bo/Documents/160945/_Color_1659125385101.43334960937500.png'

        # # data from bump area, 4th floor Levine
        # depth_filename = '/home/bo/Documents/0809_data/_Depth_1660061509679.10620117187500.csv'
        # img_name = '/home/bo/Documents/0809_data/_Color_1660061509679.12133789062500.png'

        # # data from bump area, 4th floor Levine, sofa are removed
        # depth_filename='/home/bo/Documents/0809_data_no_sofa/_Depth_1660062119778.83618164062500.csv'
        # img_name = '/home/bo/Documents/0809_data_no_sofa/_Color_1660062119778.85131835937500.png' 

        # # data from bump area, 4th floor Levine, camera elevated above table
        # depth_filename = '/home/bo/Documents/0809_data_high/_Depth_1660063198540.11767578125000.csv'
        # img_name = '/home/bo/Documents/0809_data_high/_Color_1660063198540.13183593750000.png'

        # # data from the hallway on 5th floor Levine
        # depth_filename = '/home/bo/Documents/0809_5th/_Depth_1660064029512.70043945312500.csv'
        # img_name = '/home/bo/Documents/0809_5th/_Color_1660064029512.71557617187500.png'


        # # data from the hallway on 5th floor Levine
        # depth_filename = '/home/bo/Documents/0809_5th_2/_Depth_1660064475071.03491210937500.csv'
        # img_name = '/home/bo/Documents/0809_5th_2/_Color_1660064475071.04907226562500.png' 

        # depth_filename = '/home/bo/Documents/0809_grasp/_Depth_1660066204164.48754882812500.csv'
        # img_name = '/home/bo/Documents/0809_grasp/_Color_1660066204164.50268554687500.png'

        # data from 6th floor
        img_name = '/home/bo/Documents/0809_6th/_Color_1660076119478.13476562500000.png' 
        depth_filename = '/home/bo/Documents/0809_6th/_Depth_1660076119478.12084960937500.csv'

        sg = SemanticGrid(1, self.test_ds.grid_dim, self.options.heatmap_size, self.options.cell_size,
                                    spatial_labels=self.options.n_spatial_classes, object_labels=self.options.n_object_classes)
        sg_global = SemanticGrid(1, self.test_ds.global_dim, self.options.heatmap_size, self.options.cell_size,
            spatial_labels=self.options.n_spatial_classes, object_labels=self.options.n_object_classes)                       

        ### Ground Projecting Depth Value
        file = open(depth_filename, 'rb')
        depth_loaded = np.loadtxt(file,delimiter = ",")
        print('log-----')

        imageSize = (self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
        # crop depth_abs to img_size
        depth_abs = torch.tensor(depth_loaded, device='cuda')
        print("depth_abs size cuda ",depth_abs.size())
        
        # previous xs and ys
        xs, ys = torch.tensor(np.meshgrid(np.linspace(-1,1, depth_abs.shape[1]), np.linspace(1,-1,depth_abs.shape[0])), device='cuda')
        xs = xs.reshape(1,depth_abs.shape[0],depth_abs.shape[1])
        ys = ys.reshape(1,depth_abs.shape[0],depth_abs.shape[1])

        # xs, ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.test_ds.img_size[0]), np.linspace(1,-1,self.img_size[1])), device=self.device)
        # xs = xs.reshape(1,self.test_ds.img_size[0],self.test_ds.img_size[1])
        # ys = ys.reshape(1,self.test_ds.img_size[0],self.test_ds.img_size[1])

        
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
        
        input_image = Image.open(img_name)
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
        # print('pred_ego_crops_sseg.size', pred_ego_crops_sseg.size()) # torch.Size([1, 1, 27, 192, 192])

        pred_ego_crops_sseg = map_utils.update_sseg_with_occ(pred_ego_crops_sseg, ego_grid_sseg_3, self.test_ds.grid_dim)


        # Pseudo Poses
        _rel_abs_pose = torch.tensor([[0.0, 0.0, 0.0]])
        gt_waypoints_pose_coords = torch.tensor([[ 96.,  96.],\
            #[ 95.,  96.],\
        [ 88.,  83.],\
        [ 92.,  72.],\
        [103.,  61.],\
        [114.,  52.],\
        [127.,  44.],\
        [140.,  42.],\
        [152.,  50.],\
        [166.,  44.],\
        [175.,  37.]])
        # rel_abs_pose x, y, zeta
        # abs_pose_coords: half the global grid
        # get_coord_pose
        # pose_coords
        abs_pose_coords, pose_coords, abs_pose_coords = torch.tensor([[[255, 255]]]), torch.tensor([[[95, 95]]]), torch.tensor([[[255, 255]]]) 
        abs_poses = [(8.75763, -8.06625, -2.143222037147831)] # abs pose from habitat
        rel_abs_pose = torch.tensor([0.0, 0.0, 0.0])
        ltg_counter = 0
        ltg_abs_coords = torch.zeros((1, 1, 2), dtype=torch.int64).to(self.device)
        ltg_abs_coords_list = []
        y_height = 1.25
        agent_pose = (0, 0, 0)

        t, idx = 0, 0

        # # Transform the ground projected egocentric grids to geocentric using relative pose
        # geo_grid_sseg = sg_global.spatialTransformer(grid=ego_grid_sseg_3, pose=_rel_abs_pose, abs_pose=torch.tensor(abs_poses).to(self.device))
        # # step_geo_grid contains the map snapshot every time a new observation is added
        # step_geo_grid_sseg = sg_global.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
        # # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
        # step_ego_grid_sseg = sg_global.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=_rel_abs_pose, abs_pose=torch.tensor(abs_poses).to(self.device))
        # # Crop the grid around the agent at each timestep
        # step_ego_grid_maps = map_utils.crop_grid(grid=step_ego_grid_sseg, crop_size=self.test_ds.grid_dim)
        # step_ego_grid_maps = step_ego_grid_maps.unsqueeze(0)

        # # ### visualize the updated sseg map 
        # color_ego_img_segm = viz_utils.colorize_grid(pred_ego_crops_sseg, color_mapping=27)
        # im_ego_img_segm = color_ego_img_segm[0,0,:,:,:].permute(1,2,0).cpu().numpy()
        # plt.imshow(im_ego_img_segm)
        # plt.show()


        # do ground-projection, update the projected map
        # ego_grid_sseg_3 = map_utils.est_occ_from_depth([local3D_step], grid_dim=self.test_ds.global_dim, cell_size=self.test_ds.cell_size, 
        #                                                                 device=self.device, occupancy_height_thresh=self.options.occupancy_height_thresh)
        # # Transform the ground projected egocentric grids to geocentric using relative pose
        # geo_grid_sseg = sg_global.spatialTransformer(grid=ego_grid_sseg_3, pose=_rel_abs_pose, abs_pose=torch.tensor(abs_poses).to(self.device))
        # # step_geo_grid contains the map snapshot every time a new observation is added
        # step_geo_grid_sseg = sg_global.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
        # # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
        # step_ego_grid_sseg = sg_global.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=_rel_abs_pose, abs_pose=torch.tensor(abs_poses).to(self.device))
        # # Crop the grid around the agent at each timestep
        # step_ego_grid_maps = map_utils.crop_grid(grid=step_ego_grid_sseg, crop_size=self.test_ds.grid_dim)
        # step_ego_grid_maps = step_ego_grid_maps.unsqueeze(0)


        # print(self.device)

        # print("size...", step_ego_grid_maps.size())
        # step_ego_grid_maps = step_ego_grid_maps.squeeze(0).squeeze(0).to('cpu')
        # viz_utils.vis_arr(step_ego_grid_maps[2],'.','name')
        # import copy
        # temp = copy.deepcopy(step_ego_grid_maps[1,:,:])
        # step_ego_grid_maps[1] = step_ego_grid_maps[2]
        # step_ego_grid_maps[2] = temp
        # viz_utils.vis_arr(step_ego_grid_maps[2],'.','name')
        # step_ego_grid_maps = step_ego_grid_maps.unsqueeze(0).unsqueeze(0)


        step_ego_grid_maps = map_utils.crop_grid(grid=ego_grid_sseg_3, crop_size=self.test_ds.grid_dim)
        step_ego_grid_maps = step_ego_grid_maps.unsqueeze(0)
        step_ego_grid_maps = step_ego_grid_maps.to('cpu')
        pred_ego_crops_sseg = pred_ego_crops_sseg.to('cpu')

        # run goal prediction
        # instruction = 'Go forward to pass the chair, once across the chair turn right and go to the table'
        # instruction = 'Go straight for 1 meter, make a right turn for 30 degrees, then go straight for 1 meter.'
        instruction = 'Go straight all the way down to pass the chairs, once pass those chairs make a right turn and go straight.' ## this instruction gives a good output
        instruction = 'Go straight to pass the chairs, once pass those chairs make a right turn and go straight'
        instruction = 'Go straight to pass the chair, once pass the chair make a right turn and go straight'
        instruction = 'Go straight all the way down to pass the chairs, once pass those chairs make a left turn and go straight.' 
        instruction = 'Go straight all the way down to pass the chairs and the tables, once pass those chairs make a left turn and go straight.'
        instruction = 'Go straight all the way down to past the chairs, once past those chairs make a right turn and go straight.'
        # instruction = 'Go straight all the way down to pass the chairs, once pass those chairs make a right turn and go straight for one meter, then turn right and go straight.' ## this instruction gives a good output
        # instruction = 'Go straight all the way down to pass the chairs, once pass those chairs turn right ninty degrees and go straight.'
        
        
        start_pose = [0.0, 0.0, 0.0]
        pose = (0.0, 0.0, 0.0) #(0.0, 0.0, 1.57) #(10.0, 10.0, 10.0) 
        # instruction = 'Turn right and exit the bathroom into the bedroom. Once in the bedroom turn right and walk to the door leading out on your right. Once out the door turn left and walk to the top of the stairs and stop.'
        # start_pose, pose = [8.06624984741211, 3.514145851135254, -8.757630348205566], (8.75763, -8.06625, -2.143222037147831)

        # np.save('/home/bo/Desktop/ego_occ_maps_robot.npy', step_ego_grid_maps)
        # np.save('/home/bo/Desktop/ego_segm_maps_robot.npy', pred_ego_crops_sseg)

        # try running the cm2 data here
        # step_ego_grid_maps = np.load('/home/bo/Desktop/ego_occ_maps_cm2.npy')
        # pred_ego_crops_sseg = np.load('/home/bo/Desktop/ego_segm_maps_cm2.npy')
        # step_ego_grid_maps = torch.tensor(step_ego_grid_maps)
        # pred_ego_crops_sseg = torch.tensor(pred_ego_crops_sseg)
        # print(step_ego_grid_maps.size(), pred_ego_crops_sseg)

        # step_ego_grid_maps is the one dominating the problem
        pred_waypoints_pose_coords, pred_waypoints_vals, waypoints_cov, pred_maps_spatial, pred_maps_objects = self.run_goal_pred(instruction, sg=sg, ego_occ_maps=step_ego_grid_maps, 
                                                                                                                            ego_segm_maps=pred_ego_crops_sseg, start_pos=start_pose, pose=pose)
    
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
            ### visualize the predicted waypoints vs gt waypoints in gt semantic egocentric frame
            # viz_utils.show_waypoint_pred(pred_maps_objects.squeeze(0).squeeze(0), ltg=ltg_ego_coords.clone().cpu().numpy(), pose_coords=pose_coords.clone().cpu().numpy(), num_points=self.options.num_waypoints,
            #                                     pred_waypoints=pred_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_pred.png")
            viz_utils.show_waypoint_pred(pred_maps_objects.squeeze(0).squeeze(0), ltg=None, pose_coords=None, num_points=self.options.num_waypoints,
                                    pred_waypoints=pred_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_pred.png")

            # # viz_utils.show_waypoint_pred(gt_map_semantic, num_points=self.options.num_waypoints,
            # #                                     gt_waypoints=gt_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_gt.png")

            # # ### visualize the episode steps in the global geocentric frame
            # # gt_map_semantic_global, _ = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[0],
            # #                                         grid_dim=self.test_ds.global_dim, cell_size=self.options.cell_size, color_pcloud=color_pcloud, z=z)
            # # # project gt waypoints in global geocentric frame
            # # gt_waypoints_global = torch.zeros((len(gt_waypoints), 2))
            # # for k in range(len(gt_waypoints)):
            # #     point_global_coords, _ = tutils.transform_to_map_coords(sg=sg_global, position=gt_waypoints[k], abs_pose=abs_poses[0], 
            # #                                                             grid_size=self.test_ds.global_dim[0], cell_size=self.options.cell_size, device=self.device)
            # #     gt_waypoints_global[k,:] = point_global_coords.squeeze(0).squeeze(0)                                  
            # # # transform predicted waypoints in global geocentric frame
            # # pred_waypoints_global = torch.zeros((len(pred_waypoints_pose_coords), 2))
            # # for k in range(len(pred_waypoints_pose_coords)):
            # #     pred_point_global_coords = tutils.transform_ego_to_geo(pred_waypoints_pose_coords[k].unsqueeze(0).unsqueeze(0), pose_coords, abs_pose_coords, abs_poses, t)
            # #     pred_waypoints_global[k,:] = pred_point_global_coords.squeeze(0)

            # # viz_utils.show_waypoint_pred(gt_map_semantic_global, ltg=ltg_abs_coords.clone().cpu().numpy(), pose_coords=abs_pose_coords.clone().cpu().numpy(), num_points=self.options.num_waypoints,
            # #                                     pred_waypoints=pred_waypoints_global, gt_waypoints=gt_waypoints_global, savepath=save_img_dir_+"global_"+str(t)+"_waypoints.png")

            # saves predicted areas (egocentric)
            # viz_utils.vis_arr(pred_ego_crops_sseg)
            viz_utils.save_map_pred_steps(step_ego_grid_maps, pred_maps_spatial, 
                                                pred_maps_objects, pred_ego_crops_sseg, save_img_dir_, t, instruction)
            # print('img_segm.size', img_segm.size())
            viz_utils.write_tensor_imgSegm(img_segm, save_img_dir_, name="img_segm", t=t, labels=21) # img_segm is of 21 labels in deeplabv3

        ##### Action decision process #####
        pred_goal = pred_waypoints_pose_coords[-1].unsqueeze(0).unsqueeze(0)
        pred_goal_dist = torch.linalg.norm(pred_goal.clone().float()-pose_coords.float())*self.options.cell_size # distance to current predicted goal

        # # Check stopping criteria
        # if tutils.decide_stop_vln(pred_goal_dist, self.options.stop_dist) or t==self.options.max_steps-1:
        #     t+=1
        #     break

        # the goal is passed with respect to the abs_relative pose
        # additionally moved cast depth to a tensor
        # depth = torch.Tensor(depth_abs).to(self.device)
        # depth = torch.Tensor(depth_abs).to('cuda')
        depth = depth_abs.type(torch.FloatTensor).to(self.device)
        action_id = self.run_local_policy(depth=depth, goal=ltg_abs_coords.clone(),
                                                    pose_coords=abs_pose_coords.clone(), rel_agent_o=rel_abs_pose[2], step=t)
        print('action_id:', action_id)
        # if stop is selected from local policy then randomly select an action
        if action_id==0:
            action_id = random.randint(1,3)
        # Let's apply the action
        if action_id==1:
            move_around.forward()



    def pipeline_integration(self):
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
            print('abs_poses')
            print(abs_poses)
            print('ltg_abs_coords_list')
            print(ltg_abs_coords_list)
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
            # file = open(depth_filename, 'rb')
            # depth_loaded = np.loadtxt(file,delimiter = ",")
            # print('log-----')

            depth_loaded = depth_arr

            imageSize = (self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
            # crop depth_abs to img_size
            depth_abs = torch.tensor(depth_loaded, device='cuda')
            print("depth_abs size cuda ",depth_abs.size())
            
            # previous xs and ys
            xs, ys = torch.tensor(np.meshgrid(np.linspace(-1,1, depth_abs.shape[1]), np.linspace(1,-1,depth_abs.shape[0])), device='cuda')
            xs = xs.reshape(1,depth_abs.shape[0],depth_abs.shape[1])
            ys = ys.reshape(1,depth_abs.shape[0],depth_abs.shape[1])

            # xs, ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.test_ds.img_size[0]), np.linspace(1,-1,self.img_size[1])), device=self.device)
            # xs = xs.reshape(1,self.test_ds.img_size[0],self.test_ds.img_size[1])
            # ys = ys.reshape(1,self.test_ds.img_size[0],self.test_ds.img_size[1])

            
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
            # print('pred_ego_crops_sseg.size', pred_ego_crops_sseg.size()) # torch.Size([1, 1, 27, 192, 192])

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

            gt_waypoints_pose_coords = torch.tensor([[ 96.,  96.]])
            # _rel_abs_pose = torch.tensor([[0.0, 0.0, 0.0]])
            # gt_waypoints_pose_coords = torch.tensor([[ 96.,  96.],\
            #     #[ 95.,  96.],\
            # [ 88.,  83.],\
            # [ 92.,  72.],\
            # [103.,  61.],\
            # [114.,  52.],\
            # [127.,  44.],\
            # [140.,  42.],\
            # [152.,  50.],\
            # [166.,  44.],\
            # [175.,  37.]])
            # # rel_abs_pose x, y, zeta
            # # abs_pose_coords: half the global grid
            # # get_coord_pose
            # # pose_coords
            # abs_pose_coords, pose_coords, abs_pose_coords = torch.tensor([[[255, 255]]]), torch.tensor([[[95, 95]]]), torch.tensor([[[255, 255]]]) 
            # abs_poses = [(8.75763, -8.06625, -2.143222037147831)] # abs pose from habitat
            # rel_abs_pose = torch.tensor([0.0, 0.0, 0.0])
            # ltg_counter = 0
            # ltg_abs_coords = torch.zeros((1, 1, 2), dtype=torch.int64).to(self.device)
            # ltg_abs_coords_list = []
            # y_height = 1.25
            # agent_pose = (0, 0, 0)

            # t, idx = 0, 0


            step_ego_grid_maps = map_utils.crop_grid(grid=ego_grid_sseg_3, crop_size=self.test_ds.grid_dim)
            step_ego_grid_maps = step_ego_grid_maps.unsqueeze(0)
            step_ego_grid_maps = step_ego_grid_maps.to('cpu')
            pred_ego_crops_sseg = pred_ego_crops_sseg.to('cpu')

            # run goal prediction
            # instruction = 'Go forward to pass the chair, once across the chair turn right and go to the table'
            # instruction = 'Go straight for 1 meter, make a right turn for 30 degrees, then go straight for 1 meter.'
            instruction = 'Go straight all the way down to pass the chairs, once pass those chairs make a right turn and go straight.' ## this instruction gives a good output
            instruction = 'Go straight to pass the chairs, once pass those chairs make a right turn and go straight'
            instruction = 'Go straight to pass the chair, once pass the chair make a right turn and go straight'
            instruction = 'Go straight all the way down to pass the chairs, once pass those chairs make a left turn and go straight.' 
            instruction = 'Go straight all the way down to pass the chairs and the tables, once pass those chairs make a left turn and go straight.'
            instruction = 'Go straight all the way down to past the chairs, once past those chairs make a right turn and go straight.'
            # instruction = 'Go straight all the way down to pass the chairs, once pass those chairs make a right turn and go straight for one meter, then turn right and go straight.' ## this instruction gives a good output
            # instruction = 'Go straight all the way down to pass the chairs, once pass those chairs turn right ninty degrees and go straight.'
            
            
            start_pose = [0.0, 0.0, 0.0] # the start_pose should always be the origin - [0, 0, 0]
            # pose = (0.0, 0.0, 0.0) #(0.0, 0.0, 1.57) #(10.0, 10.0, 10.0) 
            # instruction = 'Turn right and exit the bathroom into the bedroom. Once in the bedroom turn right and walk to the door leading out on your right. Once out the door turn left and walk to the top of the stairs and stop.'
            # start_pose, pose = [8.06624984741211, 3.514145851135254, -8.757630348205566], (8.75763, -8.06625, -2.143222037147831)

            # np.save('/home/bo/Desktop/ego_occ_maps_robot.npy', step_ego_grid_maps)
            # np.save('/home/bo/Desktop/ego_segm_maps_robot.npy', pred_ego_crops_sseg)

            # try running the cm2 data python main.py --name=real --vln_no_map --root_path /home/bo/Desktop/VLN_desktop/aihabitat_data/ --model_exp_dir /home/bo/Desktop/VLN_desktop/VLN_rerun/models/ --save_nav_images --use_first_waypointhere
            # step_ego_grid_maps = np.load('/home/bo/Desktop/ego_occ_maps_cm2.npy')
            # pred_ego_crops_sseg = np.load('/home/bo/Desktop/ego_segm_maps_cm2.npy')
            # step_ego_grid_maps = torch.tensor(step_ego_grid_maps)
            # pred_ego_crops_sseg = torch.tensor(pred_ego_crops_sseg)
            # print(step_ego_grid_maps.size(), pred_ego_crops_sseg)

            # step_ego_grid_maps is the one dominating the problem
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
                ### visualize the predicted waypoints vs gt waypoints in gt semantic egocentric frame
                viz_utils.show_waypoint_pred(pred_maps_objects.squeeze(0).squeeze(0), ltg=ltg_ego_coords.clone().cpu().numpy(), pose_coords=pose_coords.clone().cpu().numpy(), num_points=self.options.num_waypoints,
                                                    pred_waypoints=pred_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_pred.png")
                # viz_utils.show_waypoint_pred(pred_maps_objects.squeeze(0).squeeze(0), ltg=None, pose_coords=None, num_points=self.options.num_waypoints,
                #                         pred_waypoints=pred_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_pred.png")

                # # viz_utils.show_waypoint_pred(gt_map_semantic, num_points=self.options.num_waypoints,
                # #                                     gt_waypoints=gt_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_gt.png")

                # # ### visualize the episode steps in the global geocentric frame
                # # gt_map_semantic_global, _ = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[0],
                # #                                         grid_dim=self.test_ds.global_dim, cell_size=self.options.cell_size, color_pcloud=color_pcloud, z=z)
                # # # project gt waypoints in global geocentric frame
                # # gt_waypoints_global = torch.zeros((len(gt_waypoints), 2))
                # # for k in range(len(gt_waypoints)):
                # #     point_global_coords, _ = tutils.transform_to_map_coords(sg=sg_global, position=gt_waypoints[k], abs_pose=abs_poses[0], 
                # #                                                             grid_size=self.test_ds.global_dim[0], cell_size=self.options.cell_size, device=self.device)
                # #     gt_waypoints_global[k,:] = point_global_coords.squeeze(0).squeeze(0)                                  
                # # # transform predicted waypoints in global geocentric frame
                # # pred_waypoints_global = torch.zeros((len(pred_waypoints_pose_coords), 2))
                # # for k in range(len(pred_waypoints_pose_coords)):
                # #     pred_point_global_coords = tutils.transform_ego_to_geo(pred_waypoints_pose_coords[k].unsqueeze(0).unsqueeze(0), pose_coords, abs_pose_coords, abs_poses, t)
                # #     pred_waypoints_global[k,:] = pred_point_global_coords.squeeze(0)

                # # viz_utils.show_waypoint_pred(gt_map_semantic_global, ltg=ltg_abs_coords.clone().cpu().numpy(), pose_coords=abs_pose_coords.clone().cpu().numpy(), num_points=self.options.num_waypoints,
                # #                                     pred_waypoints=pred_waypoints_global, gt_waypoints=gt_waypoints_global, savepath=save_img_dir_+"global_"+str(t)+"_waypoints.png")

                # saves predicted areas (egocentric)
                # viz_utils.vis_arr(pred_ego_crops_sseg)
                viz_utils.save_map_pred_steps(step_ego_grid_maps, pred_maps_spatial, 
                                                    pred_maps_objects, pred_ego_crops_sseg, save_img_dir_, t, instruction)
                # print('img_segm.size', img_segm.size())
                 # img_segm is of 21 labels in deeplabv3
                viz_utils.write_tensor_imgSegm(img_segm, save_img_dir_, name="img_segm", t=t, labels=21)
            ##### Action decision process #####
            pred_goal = pred_waypoints_pose_coords[-1].unsqueeze(0).unsqueeze(0)
            pred_goal_dist = torch.linalg.norm(pred_goal.clone().float()-pose_coords.float())*self.options.cell_size # distance to current predicted goal

            # # Check stopping criteria
            # if tutils.decide_stop_vln(pred_goal_dist, self.options.stop_dist) or t==self.options.max_steps-1:
            #     t+=1
            #     break

            # the goal is passed with respect to the abs_relative pose
            # additionally moved cast depth to a tensor
            # depth = torch.Tensor(depth_abs).to(self.device)
            # depth = torch.Tensor(depth_abs).to('cuda')
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
            print('---------------------------------------------------------------------------')
            t += 1


