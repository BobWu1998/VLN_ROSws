import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets.util.map_utils as map_utils


class VLNGoalPredictor_L2M(nn.Module):
    def __init__(self, 
                 bert_model,  
                 map_predictor, 
                 map_encoder_sem, 
                 attention_model_waypoints, 
                 goal_pred_model,
                 pos_loss_scale, 
                 cov_loss_scale,
                 d_model, 
                 use_first_waypoint, 
                 loss_norm, 
                 text_init_dim=768):
        super(VLNGoalPredictor_L2M, self).__init__()

        self.bert_model = bert_model
        self.text_encoder = nn.Linear(in_features=text_init_dim, out_features=d_model) # to produce same size embeddings as the map
        self.map_predictor = map_predictor
        self.map_encoder_sem = map_encoder_sem
        self.attention_model_waypoints = attention_model_waypoints
        self.goal_pred_model = goal_pred_model

        if loss_norm:
            reduction = "mean"
        else:
            reduction = "sum"

        self.position_mse_loss = nn.MSELoss(reduction=reduction)
        self.cel_loss_cov = nn.CrossEntropyLoss(reduction=reduction)

        self.pos_loss_scale = pos_loss_scale
        self.cov_loss_scale = cov_loss_scale
        self.d_model = d_model
        self.use_first_waypoint = use_first_waypoint

        if self.use_first_waypoint:
            self.conv_point1 = nn.Conv2d(d_model+1, d_model, 1, 1, padding=0)

    
    def forward(self, batch):
        #semantic_map = batch['map_semantic']
        #print(semantic_map.shape)
        #B, T, C, H, W = semantic_map.shape # batch, sequence, 1, height, width
        #semantic_map = semantic_map.view(B*T, C, H, W)

        # egocentric inputs
        step_ego_grid_maps = batch['step_ego_grid_maps']
        ego_segm_maps = batch['ego_segm_maps']
        B, T, spatial_C, H, W = step_ego_grid_maps.shape # batch, sequence, 3, height, width
        step_ego_grid_maps = step_ego_grid_maps.view(B*T, spatial_C, H, W)
        
        objects_C = ego_segm_maps.shape[2]
        ego_segm_maps = ego_segm_maps.view(B*T, objects_C, H, W)
        #print(step_ego_grid_maps.shape)
        #print(ego_segm_maps.shape)

        # Map predictor is the L2M_model_dict (ensemble)
        # L2M normally takes input 64x64 cell_size=0.1
        # Current data are 192x192 cell_size=0.05
        # We need to preprocess current data to match L2M input

        # crop the map to 6.4m x 6.4m around the agent (L2M area of prediction)
        # then resize to input L2M size
        step_ego_grid_maps_cropped = map_utils.crop_grid(grid=step_ego_grid_maps, crop_size=(128,128))
        step_ego_grid_maps_l2m_in = F.interpolate(step_ego_grid_maps_cropped, size=(64,64), mode='nearest')
        ego_segm_maps_cropped = map_utils.crop_grid(grid=ego_segm_maps, crop_size=(128,128))
        ego_segm_maps_l2m_in = F.interpolate(ego_segm_maps_cropped, size=(64,64), mode='nearest')
        
        step_ego_grid_maps_l2m_in = step_ego_grid_maps_l2m_in.view(B, T, spatial_C, 64, 64)
        ego_segm_maps_l2m_in = ego_segm_maps_l2m_in.view(B, T, objects_C, 64, 64)

        ensemble_pred_maps = []
        ensemble_raw_maps = []
        ensemble_spatial_maps = []
        input_batch = {'step_ego_grid_crops_spatial':step_ego_grid_maps_l2m_in,
                       'pred_ego_crops_sseg':ego_segm_maps_l2m_in}
                       
        for n in range(len(self.map_predictor)):
            model_pred_output = self.map_predictor[n]['predictor_model'](input_batch)
            ensemble_pred_maps.append(model_pred_output['pred_maps_objects'])
            ensemble_raw_maps.append(model_pred_output['pred_maps_raw_objects'])
            ensemble_spatial_maps.append(model_pred_output['pred_maps_spatial'])    
        ensemble_pred_maps = torch.stack(ensemble_pred_maps) # N x B x T x C x cH x cW
        ensemble_raw_maps = torch.stack(ensemble_raw_maps)
        ensemble_spatial_maps = torch.stack(ensemble_spatial_maps)
        
        ### Estimate average predictions from the ensemble
        mean_ensemble_prediction = torch.mean(ensemble_pred_maps, dim=0) # B x T x C x cH x cW
        mean_ensemble_raw = torch.mean(ensemble_raw_maps, dim=0)
        mean_ensemble_spatial = torch.mean(ensemble_spatial_maps, dim=0)

        mean_ensemble_prediction = mean_ensemble_prediction.view(B*T, objects_C, 64, 64)
        mean_ensemble_raw = mean_ensemble_raw.view(B*T, objects_C, 64, 64)
        mean_ensemble_spatial = mean_ensemble_spatial.view(B*T, spatial_C, 64, 64)

        # Resize back to original size
        pred_maps_objects = F.interpolate(mean_ensemble_prediction, size=(H,W), mode='nearest')
        pred_maps_raw_objects = F.interpolate(mean_ensemble_raw, size=(H,W), mode='nearest')
        pred_maps_spatial = F.interpolate(mean_ensemble_spatial, size=(H,W), mode='nearest')
        #print(pred_maps_objects.shape)
        #print(pred_maps_raw_objects.shape)

        # bert inputs
        tokens_tensor = batch['tokens_tensor'].long()
        segments_tensors = batch['segments_tensors'].long()
        tokens_tensor = tokens_tensor.squeeze(1) #.view(B, N)
        segments_tensors = segments_tensors.squeeze(1) #.view(B, N)
        #print(tokens_tensor.shape)
        #print(segments_tensors.shape)

        outputs = self.bert_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        #print(hidden_states[-1][0][:].shape)
        #print(hidden_states[-1][1][:].shape)

        text_feat=[]
        for b in range(tokens_tensor.shape[0]):
            text_feat.append(hidden_states[-1][b][:])
        text_feat = torch.stack(text_feat)
        #print(text_feat.shape)

        encoded_text = self.text_encoder(text_feat)
        #print("Encoded text:", encoded_text.shape)

        # replicate encoded text for number of encoded_map_in
        if T>1:
            encoded_text = encoded_text.unsqueeze(1).repeat(1, T, 1, 1)
            encoded_text = encoded_text.view(B*T, encoded_text.shape[2], encoded_text.shape[3])


        # Second encoder + attention module for predicting the waypoints
        encoded_sem_map = self.map_encoder_sem(pred_maps_raw_objects)
        map_enc_res = encoded_sem_map.shape[2]
        encoded_sem_map_in = encoded_sem_map.permute(0,2,3,1).view(encoded_sem_map.shape[0], -1, self.d_model)
        #print(encoded_sem_map_in.shape)
        dec_out_waypoints, dec_enc_attns_waypoints = self.attention_model_waypoints(enc_inputs=encoded_text, dec_inputs=encoded_sem_map_in)
        
        dec_out_waypoints = dec_out_waypoints.permute(0,2,1).view(dec_out_waypoints.shape[0], -1, map_enc_res, map_enc_res) # reshape map attn # B x 128 x 32 x 32
        #print(dec_out_waypoints.shape)

        if self.use_first_waypoint:
            #print(batch['goal_heatmap'].shape) # B x T x num_waypoints x 48 x 48
            point_heatmap = batch['goal_heatmap'][:,:,0,:,:] # B x T x 1 x 48 x 48
            #print(point_heatmap.shape)
            point_heatmap = F.interpolate(point_heatmap, size=(dec_out_waypoints.shape[2], dec_out_waypoints.shape[3]), mode='nearest')
            #print(point_heatmap.shape)
            point_heatmap = point_heatmap.view(B*T, point_heatmap.shape[2], point_heatmap.shape[3]).unsqueeze(1)
            pred_in = torch.cat((dec_out_waypoints, point_heatmap), dim=1)
            pred_in = self.conv_point1(pred_in)
        else:
            pred_in = dec_out_waypoints
        #print("Pred in:", pred_in.shape)

        waypoints_heatmaps, waypoints_cov_raw = self.goal_pred_model(pred_in)
        #print(waypoints_heatmaps.shape)
        # get the prob distribution over uncovered/covered for each waypoint
        waypoints_cov = F.softmax(waypoints_cov_raw, dim=2)

        pred_maps_raw_objects = pred_maps_raw_objects.view(B,T,objects_C,H,W)
        pred_maps_objects = pred_maps_objects.view(B,T,objects_C,H,W)

        output = {#'pred_maps_raw_spatial':pred_maps_raw_spatial,
                  'pred_maps_raw_objects':pred_maps_raw_objects,
                  'pred_maps_spatial':pred_maps_spatial,
                  'pred_maps_objects':pred_maps_objects,
                  'pred_waypoints':waypoints_heatmaps,
                  'waypoints_cov_raw':waypoints_cov_raw,
                  'waypoints_cov':waypoints_cov,
                  #'dec_enc_attns_map':dec_enc_attns,
                  'dec_enc_attns_waypoints':dec_enc_attns_waypoints
                  }

        # ** Look at the masking in the attention module again, perhaps pass the masks for the bert features
        # ** Think of converting the input to polar coordinates before passing into the network
        # ** Reconsider about keeping the waypoint coverage estimation

        #return waypoints_heatmaps, waypoints_cov_raw, waypoints_cov, dec_enc_attns_waypoints
        return output


    def coverage_loss(self, pred_output, input_batch):
        waypoints_cov_raw = pred_output['waypoints_cov_raw']
        gt_waypoints_cov = input_batch['covered_waypoints'].long()
        B, T, num_waypoints = gt_waypoints_cov.shape
        gt_waypoints_cov = gt_waypoints_cov.view(B*T, num_waypoints)

        if self.use_first_waypoint:
            gt_waypoints_cov = gt_waypoints_cov[:,1:]
            num_waypoints = num_waypoints-1 # needed for reshaping in the loss
        #print(gt_waypoints_cov.shape)
        #print(waypoints_cov_raw.shape)
        cov_loss = self.cel_loss_cov(input=waypoints_cov_raw.contiguous().view(B*T*num_waypoints,2), 
                                     target=gt_waypoints_cov.contiguous().view(B*T*num_waypoints))

        cov_err = cov_loss.clone().detach()
        output = {'cov_loss': cov_loss*self.cov_loss_scale, 'cov_error': cov_err}
        return output


    def position_loss(self, pred_output, input_batch):
        pred_waypoints = pred_output['pred_waypoints']
        gt_waypoints = input_batch['goal_heatmap'] # B x T x num_waypoints x h x w
        visible_waypoints = input_batch['visible_waypoints'] # B x T x num_waypoints
        #print("Visible:", visible_waypoints[0,2,:])

        #if len(gt_waypoints.shape) > 4:
        B, T, num_waypoints, cH, cW = gt_waypoints.shape
        gt_waypoints = gt_waypoints.view(B*T, num_waypoints, cH, cW)
        visible_waypoints = visible_waypoints.view(B*T, num_waypoints)
        visible_waypoints = visible_waypoints.unsqueeze(2).unsqueeze(2).repeat(1, 1, cH, cW)

        if self.use_first_waypoint:
            gt_waypoints = gt_waypoints[:,1:,:,:]
            visible_waypoints = visible_waypoints[:,1:,:,:]

        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(10,5))
        #for k in range(num_waypoints-1):
            #plt.imshow(gt_waypoints.cpu().numpy()[0,k,:,:])
        #plt.imshow(pred_waypoints.detach().cpu().numpy()[0,4,:,:])
        #plt.show()
        #plt.savefig("pred_w_.png", bbox_inches='tight', pad_inches=0, dpi=100)
        #plt.close()        
        #plt.imshow(visible_waypoints.cpu().numpy()[2,7,:,:])
        #plt.show()

        # Mask waypoints that are not visible in each example
        # pred waypoints is already shaped B*T x num_waypoints x h x w
        pred_waypoints = pred_waypoints*visible_waypoints
        gt_waypoints = gt_waypoints*visible_waypoints # non visible gt heatmaps should anyway be empty

        #print(gt_waypoints.shape)
        #print(pred_waypoints.shape)
        loss = self.position_mse_loss(pred_waypoints, gt_waypoints)
        err = loss.clone().detach()
        output = {'position_loss': loss*self.pos_loss_scale, 'position_error': err}
        return output
