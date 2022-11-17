
import torch
import torch.nn as nn
import os
from models.networks import MapEncoder, MapAttention, StackedHourglass, StackedHourglassGoal, ResNetUNetGoalPred, ResNetUNetHierarchical, ResNetUNetAttn
from .holistic import HolisticWaypointPredictor
from .vln_goal_predictor import VLNGoalPredictor
from .vln_no_map_goal_predictor import VLNGoalPredictor_UnknownMap
from .vln_no_map_goal_predictor_ext import VLNGoalPredictor_UnknownMap_Ext
from .vln_goal_predictor_L2M import VLNGoalPredictor_L2M
from transformers import BertModel
#from models.predictors import get_predictor_from_options
from models.predictors import MapPredictorHier
import test_utils as tutils


def load_bert(options):
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    # bert_model = BertModel.from_pretrained(options.root_path+'bert_model/', output_hidden_states=True)
    #bert_model.save_pretrained('/home/ggeorgak/bert_model/')
    print("Loaded BERT model!")
    
    if options.finetune_bert_last_layer:
        for param in bert_model.named_parameters():
            #print(i, param[0], param[1].shape)
            if param[0] == 'pooler.dense.weight' or param[0] == 'pooler.dense.bias': # keep the last layers as requires_grad=True
                continue
            param[1].requires_grad = False
        #for param in bert_model.parameters():
        #    print(param.shape, param.requires_grad)
    return bert_model


def load_L2M(options):
    ensemble_exp = os.listdir(options.ensemble_dir) # ensemble_dir should be a dir that holds multiple experiments
    ensemble_exp.sort() # in case the models are numbered put them in order
    L2M_models_dict = {}
    for n in range(options.ensemble_size):
        segmentation_model = ResNetUNetHierarchical(out1_n_class=options.n_spatial_classes, out2_n_class=options.n_object_classes, without_attn=True)
        L2M_models_dict[n] = {'predictor_model': MapPredictorHier(segmentation_model=segmentation_model, map_loss_scale=options.map_loss_scale, with_img_segm=True) }
        L2M_models_dict[n]['predictor_model'] = nn.DataParallel(L2M_models_dict[n]['predictor_model'])

        checkpoint_dir = options.ensemble_dir + "/" + ensemble_exp[n]
        latest_checkpoint = tutils.get_latest_model(save_dir=checkpoint_dir)
        print("Model L2M", n, "loading checkpoint", latest_checkpoint)
        L2M_models_dict[n] = tutils.load_model(models=L2M_models_dict[n], checkpoint_file=latest_checkpoint)
        
        for param in L2M_models_dict[n]['predictor_model'].parameters():
            param.requires_grad = False
    
    return L2M_models_dict


def get_model_from_options(options):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if options.with_rgb_maps:
        channels_in = 4
    else:
        channels_in = 1

    if options.use_first_waypoint:
        out_channels = options.num_waypoints-1
    else:
        out_channels = options.num_waypoints

    if options.vln:
        # Model that assumes egocentric ground-truth semantic map is given
        bert_model = load_bert(options)

        map_encoder = MapEncoder(n_channel_in=channels_in, n_channel_out=options.d_model)
        attention_model = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)

        if options.hourglass_model:
            goal_pred_model = StackedHourglassGoal(out_channels=out_channels, n=options.n_hourglass_layers, 
                                                                    d_model=options.d_model, with_lstm=options.with_lstm) # in_channels is basically d_model
        else:
            goal_pred_model = ResNetUNetGoalPred(n_channel_in=options.d_model, n_class_out=out_channels, with_lstm=options.with_lstm)
        
        return VLNGoalPredictor(map_encoder=map_encoder,
                                bert_model=bert_model,
                                attention_model=attention_model,
                                goal_pred_model=goal_pred_model,
                                pos_loss_scale=options.position_loss_scale,
                                cov_loss_scale=options.cov_loss_scale,
                                loss_norm=options.loss_norm,
                                d_model=options.d_model,
                                use_first_waypoint=options.use_first_waypoint)

    elif options.vln_no_map:
        # Model that performs hallucination in addition to waypoint prediction
        bert_model = load_bert(options)
        
        if options.without_attn_1:
            map_encoder = None
            attention_model_map_prediction = None
        else:
            # Attention for map prediction
            map_encoder = MapEncoder(n_channel_in=options.n_spatial_classes, n_channel_out=options.d_model)
            attention_model_map_prediction = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)

        # Hierarchical prediction model
        map_predictor = ResNetUNetHierarchical(out1_n_class=options.n_spatial_classes, out2_n_class=options.n_object_classes, without_attn=options.without_attn_1)

        # Second encoder + attention for waypoint prediction
        map_encoder_sem = MapEncoder(n_channel_in=options.n_object_classes, n_channel_out=options.d_model)
        attention_model_waypoints = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)

        goal_pred_model = ResNetUNetGoalPred(n_channel_in=options.d_model, n_class_out=out_channels, with_lstm=options.with_lstm)
        
        return VLNGoalPredictor_UnknownMap(map_encoder=map_encoder,
                                        bert_model=bert_model,
                                        attention_model_map_prediction=attention_model_map_prediction,
                                        map_predictor=map_predictor,
                                        map_encoder_sem=map_encoder_sem,
                                        attention_model_waypoints=attention_model_waypoints,
                                        goal_pred_model=goal_pred_model,
                                        map_loss_scale=options.map_loss_scale,
                                        pos_loss_scale=options.position_loss_scale,
                                        cov_loss_scale=options.cov_loss_scale,
                                        loss_norm=options.loss_norm,
                                        d_model=options.d_model,
                                        use_first_waypoint=options.use_first_waypoint)

    elif options.vln_no_map_ext:
        # Extended model with larger model and maps
        bert_model = load_bert(options)

        ## Block 1 of attention for map prediction
        map_encoder_spatial = MapEncoder(n_channel_in=options.n_spatial_classes, n_channel_out=options.d_model)
        attention_model_map_prediction_spatial = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)
        spatial_map_predictor = ResNetUNetAttn(n_channel_in=options.n_spatial_classes, n_class_out=options.n_spatial_classes)

        # ** Pass the conv features to each Unet, not sure about the first Unet
        # ** use an intermediate feature representation from the image segmentation as input to the second UNet 
        # ** technically we can have N blocks of attention for map prediction that are stacked together
        # ** check positional encodings
        # ** start generating the data to pretrain the high resolution L2M model and the image segmentation model

        ## Block 2 of attention for map prediction
        map_encoder_objects = MapEncoder(n_channel_in=options.n_spatial_classes, n_channel_out=options.d_model)
        attention_model_map_prediction_objects = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)        
        #objects_map_predictor = ResNetUNetAttn(n_channel_in=options.n_spatial_classes+1, n_class_out=options.n_object_classes)
        objects_map_predictor = ResNetUNetAttn(n_channel_in=options.n_spatial_classes+options.n_object_classes, n_class_out=options.n_object_classes)


        ## Block of attention for path prediction
        map_encoder_path = MapEncoder(n_channel_in=options.n_object_classes, n_channel_out=options.d_model)
        attention_model_path = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)
        #if high_res_heatmaps=True, then keypoint heatmaps are 96x96, if False then 48x48
        path_pred_model = ResNetUNetGoalPred(n_channel_in=options.d_model, n_class_out=out_channels, with_lstm=options.with_lstm, high_res_heatmaps=True)


        return VLNGoalPredictor_UnknownMap_Ext(bert_model=bert_model,
                                                map_encoder_spatial=map_encoder_spatial,
                                                map_encoder_objects=map_encoder_objects,
                                                attention_model_map_prediction_spatial=attention_model_map_prediction_spatial,
                                                attention_model_map_prediction_objects=attention_model_map_prediction_objects,
                                                spatial_map_predictor=spatial_map_predictor,
                                                objects_map_predictor=objects_map_predictor,
                                                map_encoder_path=map_encoder_path,
                                                attention_model_path=attention_model_path,
                                                path_pred_model=path_pred_model,
                                                map_loss_scale=options.map_loss_scale,
                                                pos_loss_scale=options.position_loss_scale,
                                                cov_loss_scale=options.cov_loss_scale,
                                                loss_norm=options.loss_norm,
                                                d_model=options.d_model,
                                                use_first_waypoint=options.use_first_waypoint)


    elif options.vln_L2M:
        # Model that uses L2M (frozen) to predict the map which is passed as input to the waypoints predictor
        bert_model = load_bert(options)
        L2M_models_dict = load_L2M(options) # map predictor

        map_encoder_sem = MapEncoder(n_channel_in=options.n_object_classes, n_channel_out=options.d_model)
        attention_model_waypoints = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)

        goal_pred_model = ResNetUNetGoalPred(n_channel_in=options.d_model, n_class_out=out_channels, with_lstm=options.with_lstm)

        return VLNGoalPredictor_L2M(bert_model=bert_model,
                                    map_predictor=L2M_models_dict,
                                    map_encoder_sem=map_encoder_sem,
                                    attention_model_waypoints=attention_model_waypoints,
                                    goal_pred_model=goal_pred_model,
                                    pos_loss_scale=options.position_loss_scale,
                                    cov_loss_scale=options.cov_loss_scale,
                                    loss_norm=options.loss_norm,
                                    d_model=options.d_model,
                                    use_first_waypoint=options.use_first_waypoint)


    else: 
        # Model predicting waypoints on the geocentric maps
        map_encoder = MapEncoder(n_channel_in=channels_in, n_channel_out=options.d_model)
        attention_model = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                            n_heads=options.n_heads, n_layers=options.n_layers, device=device)
        waypoint_pred_model = StackedHourglass(out_channels=out_channels, n=options.n_hourglass_layers, d_model=options.d_model, with_lstm=options.with_lstm) # in_channels is basically d_model

        return HolisticWaypointPredictor(map_encoder=map_encoder, 
                                        attention_model=attention_model,
                                        waypoint_pred_model=waypoint_pred_model,
                                        pos_loss_scale=options.position_loss_scale,
                                        head_loss_scale=options.heading_loss_scale,
                                        loss_norm=options.loss_norm,
                                        d_model=options.d_model,
                                        concat_inputs=options.concatenate_inputs,
                                        with_rgb_maps=options.with_rgb_maps,
                                        use_first_waypoint=options.use_first_waypoint)