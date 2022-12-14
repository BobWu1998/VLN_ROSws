
import numpy as np
import os
import math
import torch
import quaternion
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import torch.nn.functional as F
import gzip
import json
import random

#############################
### Functions for RxR dataloader
#############################

def read_json_lines(filepath):
    data = []
    with gzip.open(filepath, "rt") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def locs_to_heatmaps(keypoints, img_size, out_size, sigma=1):
    x_scale = out_size[0]/img_size[0]
    y_scale = out_size[1]/img_size[1]
    x = torch.arange(0, out_size[1], dtype=torch.float32)
    y = torch.arange(0, out_size[0], dtype=torch.float32)
    yg, xg = torch.meshgrid(y,x) #np.meshgrid(y,x, indexing='ij')

    gaussian_hm = torch.zeros((keypoints.shape[0], out_size[0], out_size[1]))
    for i,keypoint in enumerate(keypoints):
        kp_x = keypoint[0] * x_scale
        kp_y = keypoint[1] * y_scale
        gaussian_hm[i,:,:] = torch.exp(-((xg-kp_x)**2+(yg-kp_y)**2)/(2*sigma**2))
    return gaussian_hm


def heatmaps_to_locs(heatmaps, thresh=0):
    vals, uv = torch.max(heatmaps.view(heatmaps.shape[0], 
                                    heatmaps.shape[1], 
                                    heatmaps.shape[2]*heatmaps.shape[3]), 2)
    # zero out entries below the detection threshold
    #thresh = self.detection_thresh
    #if no_thresh:
    #    thresh = 0
    uv *= (vals > thresh).type(torch.long)
    vals *= (vals > thresh).type(torch.long)
    rows = uv / heatmaps.shape[3]
    cols = uv % heatmaps.shape[3]
    return torch.stack([cols, rows], 2).cpu().type(torch.float), vals


#def pck(gt_heatmaps, pred_heatmaps, dist_thresh=10):
def pck(gt_heatmaps, pred_heatmaps, visible=None):
    dist_thresh = gt_heatmaps.shape[2] / 5 #10 # PCK is @ 1.92m
    gt_locs, _ = heatmaps_to_locs(gt_heatmaps)
    pred_locs, _ = heatmaps_to_locs(pred_heatmaps)
    ##visible_keypoints = (gt_locs[:,:,0] > 0)
    if visible is not None:
        visible = (visible > 0)
        return 100 * torch.mean((torch.sqrt(torch.sum((gt_locs - pred_locs) ** 2, dim=-1))[visible] < dist_thresh).type(torch.float))
    else:
        return 100 * torch.mean((torch.sqrt(torch.sum((gt_locs - pred_locs) ** 2, dim=-1)) < dist_thresh).type(torch.float))


def angle_diff(target_theta, pred_theta):
    return torch.abs(torch.atan2(torch.sin(target_theta-pred_theta), torch.cos(target_theta-pred_theta)))


def get_agent_location(pose):
    # Converting pose from RxR to our convention following utils.get_sim_location()
    # Here t[1] is not useful because the height does not correspond to the ground
    R = pose[:3,:3]
    t = pose[:3,3]
    x = -t[2]
    y = -t[0]
    height = t[1]
    quad = quaternion.from_rotation_matrix(R)
    axis = quaternion.as_euler_angles(quad)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(quad)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(quad)[1]
    if o > np.pi:
        o -= 2 * np.pi
    pose = x, y, o
    return pose, height


def get_episode_pose(pose):
    # Convert pose from RxR to the episodes convention in R2R_VLNCE
    R = pose[:3,:3]
    t = pose[:3,3]
    quad = quaternion.from_rotation_matrix(R)
    return t, quad


def filter_pose_trace(poses_traced, return_idx=False):
    # There are a lot of almost duplicates in the pose trace
    # Keep next pose only if it's sufficiently different from previous
    poses_valid = []
    heights = []
    idx = []
    for i in range(poses_traced.shape[0]):
        pose_traced_0 = poses_traced[i]
        pose0, h = get_agent_location(pose_traced_0)
        pose0 = torch.tensor(pose0)
        if i==0:
            poses_valid.append(pose0)
            heights.append(h)
            idx.append(i)
        else:
            last_valid = poses_valid[-1]
            dist = torch.linalg.norm(pose0[:2] - last_valid[:2])
            angle_diff = wrap_angle(pose0[2]-last_valid[2])
            if dist > 0.1 or angle_diff > 0.1:
                poses_valid.append(pose0)
                #print(pose0[2])
                heights.append(h)
                idx.append(i)
        #print("Agent pose:", pose0)
    #if len(poses_valid) < 2:
    #    pose0, h = get_agent_location(poses_traced[-1])
    #    poses_valid.append(torch.tensor(pose0))
    #    heights.append(h)
    if return_idx:
        return poses_valid, heights, idx
    return poses_valid, heights


def sample_waypoints(poses_valid, interval=0.1, num_waypoints=10):
    # Sample points between the pose coords
    # First interpolate between the points, then select K
    # Note: This function selects non-plausible poses that cannot be used in the simulation.
    #       Therefore this function is only used to generate the ground-truth for the first problem setup.
    #       To keep only valid poses, then omit this function.
    # interval: Distance between points during interpolation
    waypoints_all = torch.tensor([])
    for i in range(len(poses_valid)-1):
        p0 = poses_valid[i]
        p1 = poses_valid[i+1]
        n_y = torch.abs(p0[0]-p1[0])/interval
        n_x = torch.abs(p0[1]-p1[1])/interval
        if n_x >= n_y:
            num_p = int(n_x)
        else:
            num_p = int(n_y)
        interp_x = torch.linspace(p0[1], p1[1], num_p+1)
        interp_y = torch.linspace(p0[0], p1[0], num_p+1)
        interp_o = torch.linspace(p0[2], p1[2], num_p+1) # dummy angle interpolation
        if i < len(poses_valid)-2: # remove last point except in the last iteration
            interp_x = interp_x[:-1]
            interp_y = interp_y[:-1]
            interp_o = interp_o[:-1]
        for k in range(len(interp_o)):
            interp_o[k] = wrap_angle(interp_o[k])         
        
        points_tmp = torch.stack((interp_y, interp_x, interp_o), dim=1)

        if i==0:
            waypoints_all = points_tmp.clone()
        else:
            waypoints_all = torch.cat((waypoints_all, points_tmp.clone()), dim=0)
    #print("all:", waypoints_all.shape)

    if waypoints_all.shape[0]<2:
        return None

    # Sample k waypoints to use as ground-truth
    k = math.ceil(waypoints_all.shape[0] / (num_waypoints-1))
    #print("k", k)
    waypoints = waypoints_all[::k] # need to verify this always gives num_waypoints
    waypoints = torch.cat((waypoints, waypoints_all[-1].view(1,-1)), dim=0) # add last element
    
    while waypoints.shape[0] < num_waypoints:
        rand_idx = random.randint(1, len(waypoints_all)-1)
        #waypoints = torch.cat((waypoints, waypoints_all[rand_idx].view(1,-1)), dim=0)
        waypoints = torch.cat((waypoints, waypoints_all[-1].view(1,-1)), dim=0) # add last element again
    while waypoints.shape[0] > num_waypoints:
        mid = int(waypoints.shape[0]/2) # remove mid element
        waypoints = torch.cat((waypoints[:mid], waypoints[mid+1:]), dim=0)
    return waypoints


def wrap_angle(o):
    # convert angle to -pi,pi range
    if o < -math.pi:
        o += 2*math.pi
    if o > math.pi:
        o -= 2*math.pi
    return o

#############################
#############################

def add_uniform_noise(tensor, a, b):
    return tensor + torch.FloatTensor(tensor.shape).uniform_(a, b).to(tensor.device)

def add_gaussian_noise(tensor, mean, std):
    return tensor + torch.randn(tensor.size()).to(tensor.device) * std + mean

def euclidean_distance(position_a, position_b):
    return np.linalg.norm(position_b - position_a, ord=2)


def preprocess_img(img, cropSize, pixFormat, normalize):
    img = img.permute(2,0,1).unsqueeze(0).float()
    img = F.interpolate(img, size=cropSize, mode='bilinear', align_corners=True)
    img = img.squeeze(0)
    if normalize:
        img = img / 255.0
    return img


# normalize code from habitat lab:
# obs = (obs - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
def unnormalize_depth(depth, min, max):
    return (depth * (max - min)) + min


def get_entropy(pred):
    log_predictions = torch.log(pred)
    mul_map = -pred*log_predictions
    return torch.sum(mul_map, dim=2, keepdim=True) # B x T x 1 x cH x cW


def add_to_quarternion(rotation, angle):
    # rotation is quarternion from "start_rotation" of episode
    # angle in radians
    # Returns quarternion
    rotation = np.quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
    euler_angles = quaternion.as_euler_angles(rotation)
    euler_angles[1] += angle
    euler_angles[1] = wrap_angle(euler_angles[1])
    return quaternion.from_euler_angles(euler_angles)


def get_sim_location(agent_state):
    x = -agent_state.position[2]
    y = -agent_state.position[0]
    height = agent_state.position[1]
    axis = quaternion.as_euler_angles(agent_state.rotation)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    pose = x, y, o
    return pose, height


def get_rel_pose(pos2, pos1):
    x1, y1, o1 = pos1
    if len(pos2)==2: # if pos2 has no rotation
        x2, y2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        return dx, dy
    else:
        x2, y2, o2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        do = o2 - o1
        if do < -math.pi:
            do += 2 * math.pi
        if do > math.pi:
            do -= 2 * math.pi
        return dx, dy, do


def load_scene_pcloud(preprocessed_scenes_dir, scene_id, n_object_classes):
    pcloud_path = preprocessed_scenes_dir+scene_id+'_pcloud.npz'
    if not os.path.exists(pcloud_path):
        raise Exception('Preprocessed point cloud for scene', scene_id,'not found!')

    data = np.load(pcloud_path)
    x = data['x']
    y = data['y']
    z = data['z']
    label_seq = data['label_seq']
    data.close()

    label_seq[ label_seq<0.0 ] = 0.0
    # Convert the labels to the reduced set of categories
    label_seq_spatial = label_seq.copy()
    label_seq_objects = label_seq.copy()
    for i in range(label_seq.shape[0]):
        curr_lbl = label_seq[i,0]
        label_seq_spatial[i] = viz_utils.label_conversion_40_3[curr_lbl]
        label_seq_objects[i] = viz_utils.label_conversion_40_27[curr_lbl]
    return (x, y, z), label_seq_spatial, label_seq_objects


def load_scene_color(preprocessed_scenes_dir, scene_id):
    # loads the rgb information of the map
    color_path = preprocessed_scenes_dir+scene_id+'_color.npz'
    if not os.path.exists(color_path):
        raise Exception('Preprocessed color for scene', scene_id,'not found!')

    data = np.load(color_path)
    #print(list(data.keys()))
    r = data['r']
    g = data['g']
    b = data['b']
    color_pcloud = np.stack((r,g,b)) # 3 x Npoints
    return color_pcloud


def depth_to_3D(depth_obs, img_size, xs, ys, inv_K):

    # depth = depth_obs[...,0].reshape(1, img_size[0], img_size[1])

    depth = depth_obs.reshape(1, depth_obs.shape[0], depth_obs.shape[1])

    # Unproject
    # negate depth as the camera looks along -Z
    # SPEEDUP - create ones in constructor
    print(xs.size(), depth.size(), torch.mul(xs, depth).size())
    # print(xs)
    xys = torch.vstack((torch.mul(xs, depth) , torch.mul(ys, depth), -depth, torch.ones(depth.shape, device='cuda'))) # 4 x 128 x 128
    xys = xys.reshape(4, -1)
    xy_c0 = torch.matmul(inv_K, xys)

    # SPEEDUP - don't allocate new memory, manipulate existing shapes
    local3D = torch.zeros((xy_c0.shape[1],3), dtype=torch.float32, device='cuda')
    local3D[:,0] = xy_c0[0,:]
    local3D[:,1] = xy_c0[1,:]
    local3D[:,2] = xy_c0[2,:]

    return local3D



def run_img_segm(model, input_batch, object_labels, crop_size, cell_size, xs, ys, inv_K, points2D_step):
    # this semantic segmtation is adapted from deeplabv3
    # currently supports batch size of 1 and time sequence of 1
    label_conversion_deeplabv3_to_27 = {
        9: 1, # chair
        11: 3, # table
        18: 5, # sofa
        20: 10 # tv_monitor
        # 15: 3 #testing purpose, map people to table
    }

    if torch.cuda.is_available():
        img_input_batch = input_batch['images'].to('cuda').squeeze(0)
        model.to('cuda')
    # print(img_input_batch.size())
    with torch.no_grad():
        pred_img_segm = model(img_input_batch)['out'][0]
        # output = model(segm_batch["images"])['out'][0]
    img_labels = pred_img_segm.argmax(0) # get the most likely prediction 
    img_labels = img_labels.cpu().detach().numpy() # convert tensor to numpy
    img_labels = torch.tensor(np.vectorize(label_conversion_deeplabv3_to_27.get)(img_labels, 16)) # 16 other
    img_labels.to('cuda') # move back to gpu

    img_labels = img_labels.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # ground-project the predicted segm
    depth_imgs = input_batch['depth_imgs']
    pred_ego_crops_sseg = torch.zeros((depth_imgs.shape[0], depth_imgs.shape[1], object_labels,
                                                    crop_size[0], crop_size[1]), dtype=torch.float32).to(depth_imgs.device)
    for b in range(depth_imgs.shape[0]): # batch size

        points2D = []
        local3D = []
        for i in range(depth_imgs.shape[1]): # sequence

            depth = depth_imgs[b,i,:,:,:].permute(1,2,0)
            local3D_step = depth_to_3D(depth, img_size=(depth.shape[0],depth.shape[1]), xs=xs, ys=ys, inv_K=inv_K)

            points2D.append(points2D_step)
            local3D.append(local3D_step)

        pred_ssegs = img_labels[b,:,:,:,:]

        # use crop_size directly for projection
        pred_ego_crops_sseg_seq = map_utils.ground_projection(points2D, local3D, pred_ssegs,
                                                            sseg_labels=object_labels, grid_dim=crop_size, cell_size=cell_size)
        pred_ego_crops_sseg[b,:,:,:,:] = pred_ego_crops_sseg_seq
    return pred_ego_crops_sseg, pred_img_segm.unsqueeze(0) # the second output has not been mapped and is still in 21 classes of deeplabv3 #pred_img_segm.argmax(0).unsqueeze(0)#pred_img_segm['pred_segm'].squeeze(0)


# Taken from: https://github.com/pytorch/pytorch/issues/35674
def unravel_index(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


# invalid ids found in RxR guide_data episodes
def get_invalid_ids(split):
    if split == "val_unseen":
    # certain instruction ids cannot produce valid waypoints, not sure why # the following are for val_unseen
        invalid_instruction_ids = [9619, 11831, 13079, 21087, 35329, 41651, 54691, 67961, 72416, 78843, 89738, 116437, 
                                    121531, 123148, 125417, 104652, 16840, 96686, 33245, 105708, 21366, 75556]
    elif split == "train":
        invalid_instruction_ids = [25, 2964, 3285, 3285, 5759, 5759, 6420, 6420, 105606, 31170, 39104, 63599, 102712, 83073,
        8270, 8270, 8953, 8953, 10204, 10204, 11424, 11424, 11977, 11977, 13554, 13554, 18352, 18352, 18781, 18781, 
        19405, 19405, 19724, 19724, 20548, 20548, 20903, 20903, 22445, 22445, 24724, 24724, 25344, 25344, 26050, 26050, 
        27271, 27271, 27411, 27411, 27535, 27535, 27888, 27888, 30838, 31689, 31689, 33628, 33628, 36031, 36031, 41194, 
        41194, 43576, 43576, 45061, 45061, 45670, 45670, 47119, 47119, 48518, 48518, 48726, 48726, 48981, 48981, 51432, 
        53883, 53883, 54516, 54516, 55210, 55210, 57140, 57140, 57446, 57446, 58336, 58336, 59667, 59667, 62304, 62304, 
        62880, 62880, 66260, 66260, 66714, 66714, 67798, 67798, 70305, 70305, 71149, 71149, 71642, 72196, 72196, 72505, 
        72505, 72799, 72799, 73175, 73944, 73944, 75910, 75910, 76132, 76132, 77485, 77485, 81039, 82949, 82949, 83690, 
        87236, 87236, 89677, 89677, 90115, 90115, 90411, 90411, 93914, 93914, 94725, 95018, 95018, 95231, 95231, 95534, 
        98252, 98252, 99719, 99719, 100841, 100841, 105397, 105397, 106097, 106097, 107274, 107274, 107740, 107740, 107845, 
        107845, 108576, 108576, 108804, 108804, 109230, 109230, 110092, 110092, 111855, 111855, 110502, 7249, 114819, 114819, 
        115328, 115328, 116651, 116651, 116914, 116914, 118056, 118056, 119338, 119338, 120868, 120868, 120946, 120946, 121121, 
        121121, 122054, 122054, 122320, 122320, 123053, 123053, 123400, 123400, 125637, 125637]
    else:
        invalid_instruction_ids = []
    return invalid_instruction_ids
