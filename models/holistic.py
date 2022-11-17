import torch
import torch.nn as nn
import torch.nn.functional as F


class HolisticWaypointPredictor(nn.Module):
    def __init__(self, map_encoder, attention_model, waypoint_pred_model, pos_loss_scale, 
                                        head_loss_scale, d_model, loss_norm, concat_inputs, with_rgb_maps, use_first_waypoint, text_init_dim=768):
        super(HolisticWaypointPredictor, self).__init__()

        self.text_encoder = nn.Linear(in_features=text_init_dim, out_features=d_model) # to produce same size embeddings as the map
        self.map_encoder = map_encoder
        self.attention_model = attention_model
        self.waypoint_pred_model = waypoint_pred_model

        if loss_norm:
            self.position_mse_loss = nn.MSELoss()
            self.heading_mse_loss = nn.MSELoss()
        else:
            self.position_mse_loss = nn.MSELoss(reduction="sum")
            self.heading_mse_loss = nn.MSELoss(reduction="sum")

        self.pos_loss_scale = pos_loss_scale
        self.head_loss_scale = head_loss_scale

        self.d_model = d_model
        self.concat_inputs = concat_inputs
        self.with_rgb_maps = with_rgb_maps
        self.use_first_waypoint = use_first_waypoint

        if self.use_first_waypoint:
            self.conv_point1 = nn.Conv2d(d_model+2, d_model, 1, 1, padding=0)



    def forward(self, batch):
        semantic_map = batch['map_semantic']
        text_feat = batch['text_feat']
        
        #print("Semantic map:", semantic_map.shape)
        #print("Color map:", batch['map_color'].shape)
        #print("Text feat:", text_feat.shape)

        if self.with_rgb_maps:
            map_in = torch.cat((semantic_map, batch['map_color']), dim=1)
        else:
            map_in = semantic_map

        ### Prepare the input embeddings of the map and text
        encoded_map = self.map_encoder(map_in) # B x 128 x 32 x 32
        #print("Encoded map:", encoded_map.shape)
        map_enc_res = encoded_map.shape[2]

        encoded_map_in = encoded_map.permute(0,2,3,1).view(encoded_map.shape[0], -1, self.d_model)
        #print("Encoded map in:", encoded_map_in.shape)
        #encoded_text = text_feat
        encoded_text = self.text_encoder(text_feat)
        #print("Encoded text:", encoded_text.shape)
        

        if self.concat_inputs: # uses more memory
            ### Apply self-attention on the concatenated inputs
            attn_in = torch.cat((encoded_map_in, encoded_text), dim=1)
            dec_out, dec_enc_attns = self.attention_model(enc_inputs=attn_in, dec_inputs=attn_in)
            # the dec_out dimension varies so we use adaptive pooling to bring down to map dimensions to pass in the waypoint model
            dec_out = F.adaptive_avg_pool1d(input=dec_out.permute(0,2,1), output_size=encoded_map_in.shape[1])
            dec_out = dec_out.permute(0,2,1)
            # In the concatenated case the attn is M+N x M+N (i.e. self attention between the entire input)
            # We need only N x M (N:map, M:tokens)
            dec_enc_attns = dec_enc_attns[:, :, :, :encoded_map_in.shape[1], encoded_map_in.shape[1]:]
        else:
            ### Apply attention between the map and text
            dec_out, dec_enc_attns = self.attention_model(enc_inputs=encoded_text, dec_inputs=encoded_map_in) # B x 1024 x 128
            #dec_out, dec_enc_attns = self.attention_model(enc_inputs=encoded_map_in, dec_inputs=encoded_text)
        #print("Dec out:", dec_out.shape)
        #print("Dec attn:", dec_enc_attns.shape)

        #print("Keypoint heatmaps gt:", batch['keypoint_heatmaps'].shape)
        #print("Headings gt:", batch['headings'].shape)
        ### Use the attention-augmented map embeddings to predict the waypoints
        dec_out = dec_out.permute(0,2,1).view(dec_out.shape[0], -1, map_enc_res, map_enc_res) # reshape map attn # B x 128 x 32 x 32
        #print("Reshaped map attn:", dec_out.shape)

        if self.use_first_waypoint:
            point_heatmap = batch['keypoint_heatmaps'][:,0,:,:].unsqueeze(1) # B x 1 x 64 x 64
            point_heatmap = F.interpolate(point_heatmap, size=(dec_out.shape[2], dec_out.shape[3]), mode='nearest')
            point_heading = batch['headings'][:,:,0].unsqueeze(1).unsqueeze(1) # B x 1 x 1 x 2
            point_heading = F.interpolate(point_heading, size=(dec_out.shape[2], dec_out.shape[3]), mode='nearest')
            pred_in = torch.cat((dec_out, point_heatmap, point_heading), dim=1)
            pred_in = self.conv_point1(pred_in)
        else:
            pred_in = dec_out
        #print("Input to waypoint model:", pred_in.shape)

        #keypoint_heatmaps, pred_headings = self.waypoint_pred_model(semantic_map, dec_out)
        #keypoint_heatmaps, pred_headings = self.waypoint_pred_model(encoded_map, dec_out)
        keypoint_heatmaps, pred_headings = self.waypoint_pred_model(pred_in)
        #print("Pred heatmaps 1:", keypoint_heatmaps[0].shape)
        #print("Pred heatmaps 2:", keypoint_heatmaps[1].shape)
        #pred_headings = pred_headings.squeeze(3).squeeze(2)
        #print("Pred headings:", pred_headings.shape)
        return keypoint_heatmaps, pred_headings, dec_enc_attns


    def position_loss(self, pred_waypoints, input_batch):
        #print(pred_waypoints[0].shape)
        #print(pred_waypoints[1].shape)
        #print(input_batch['keypoint_heatmaps'].shape)
        
        '''
        dummy_pred = pred_waypoints[0].squeeze(0)[0]
        dummy_gt = input_batch['keypoint_heatmaps'].squeeze(0)[0]
        #custom_loss = torch.sum((dummy_pred-dummy_gt)**2) / (dummy_pred.shape[0]*dummy_pred.shape[1])
        loss0 = self.position_mse_loss(dummy_pred, dummy_gt)
        print("Loss:", loss0)
        import datasets.util.viz_utils as viz_utils
        viz_utils.vis_heatmaps(dummy_pred, dummy_gt)
        '''
        if self.use_first_waypoint:
            gt_waypoints = input_batch['keypoint_heatmaps'][:,1:,:,:]
        else:
            gt_waypoints = input_batch['keypoint_heatmaps']
        
        #loss0 = self.position_mse_loss(pred_waypoints[0], input_batch['keypoint_heatmaps'])
        loss = self.position_mse_loss(pred_waypoints[1], gt_waypoints)
        #loss=loss0+loss1
        err = loss.clone().detach()
        output = {'position_loss': loss*self.pos_loss_scale, 'position_error': err}
        return output

    def heading_loss(self, pred_heading, input_batch):
        #print("GT angle:", input_batch['angle_headings'][0,0])
        # Recover the actual angles from predicted sin and cos - for debugging
        #print("Pred:", torch.atan2(pred_heading[0,0,0], pred_heading[0,1,0]))
        if self.use_first_waypoint:
            gt_headings = input_batch['headings'][:,:,1:]
        else:
            gt_headings = input_batch['headings']
        
        loss = self.heading_mse_loss(pred_heading, gt_headings)
        err = loss.clone().detach()
        output = {'heading_loss':loss*self.head_loss_scale, 'heading_error':err}
        return output
        

