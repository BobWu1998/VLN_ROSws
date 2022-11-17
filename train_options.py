from pytorch_utils.base_options import BaseOptions
import argparse

class TrainOptions(BaseOptions):
    """ Parses command line arguments for training
    This overwrites options from BaseOptions
    """
    def __init__(self): # pylint: disable=super-init-not-called
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        hard = self.parser.add_argument_group('Hardware')
        hard.add_argument('--fx', default=963.893/1000, help='focal length in x(m)') # 963.893
        hard.add_argument('--fy', default=963.893/1000, help='focal length in y(m)') #  963.893 or 962.998
        hard.add_argument('--u0', default=0, help='center of image(m)') # 962.998/1000
        hard.add_argument('--v0', default=0, help='center of image(m)') # 540.903/1000


        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=3600000,
                         help='Total time to run in seconds')
        gen.add_argument('--resume', dest='resume', default=False,
                         action='store_true',
                         help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=0,
                         help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                         help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                         help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        in_out = self.parser.add_argument_group('io')
        in_out.add_argument('--log_dir', default='./semantic_grid/logs', help='Directory to store logs')
        in_out.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        in_out.add_argument('--from_json', default=None,
                            help='Load options from json file instead of the command line')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=1000,
                           help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_nav_batch_size', type=int, default=1, help='Batch size during navigation test')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true',
                                   help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false',
                                   help='Don\'t shuffle training data')
        shuffle_test = train.add_mutually_exclusive_group()
        shuffle_test.add_argument('--shuffle_test', dest='shuffle_test', action='store_true',
                                  help='Shuffle testing data')
        shuffle_test.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false',
                                  help='Don\'t shuffle testing data')

        # Dataset related options
        #train.add_argument('--data_type', dest='data_type', type=str, default='train',
        #                    choices=['train', 'val'],
        #                    help='Choose which dataset to run on, valid only with --use_store')
        train.add_argument('--dataset_percentage', dest='dataset_percentage', type=float, default=1.0,
                            help='percentage of dataset to be used during training for ensemble learning')

        train.add_argument('--summary_steps', type=int, default=200,
                           help='Summary saving frequency')
        train.add_argument('--image_summary_steps', type=int, default=500,
                           help='Image summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=10000,
                           help='Chekpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=10000, help='Testing frequency')


        train.add_argument('--is_train', dest='is_train', action='store_true',
                            help='Define whether training or testing mode')

        train.add_argument('--config_train_file', type=str, dest='config_train_file',
                            default='configs/my_objectnav_mp3d_train.yaml',
                            help='path to habitat dataset train config file')

        self.parser.add_argument('--config_test_file', type=str, dest='config_test_file',
                                default='configs/my_objectnav_mp3d_test.yaml',
                                help='path to test config file -- to be used with our episodes')

        self.parser.add_argument('--config_val_file', type=str, dest='config_val_file',
                                default='configs/my_objectnav_mp3d_val.yaml',
                                help='path to habitat dataset val config file')

        self.parser.add_argument('--ensemble_dir', type=str, dest='ensemble_dir', default=None,
                                help='Path containing the experiments comprising the ensemble')
        
        self.parser.add_argument('--model_exp_dir', type=str, dest='model_exp_dir', default=None,
                                help='Path for experiment containing the model used for testing')                        

        self.parser.add_argument('--n_spatial_classes', type=int, default=3, dest='n_spatial_classes',
                                help='number of categories for spatial prediction')
        self.parser.add_argument('--n_object_classes', type=int, default=27, dest='n_object_classes',
                                choices=[18,27], help='number of categories for object prediction')

        self.parser.add_argument('--global_dim', type=int, dest='global_dim', default=512)
        self.parser.add_argument('--grid_dim', type=int, default=192, dest='grid_dim', # 512 192
                                    help='Semantic grid size (grid_dim, grid_dim)')
        self.parser.add_argument('--cell_size', type=float, default=0.05, dest="cell_size", # 0.1
                                    help='Physical dimensions (meters) of each cell in the grid')
        self.parser.add_argument('--crop_size', type=int, default=64, dest='crop_size',
                                    help='Size of crop around the agent')

        self.parser.add_argument('--img_size', dest='img_size', type=int, default=[640,480])#[1280, 720])#[1280, 720]) #[256,256])#
        self.parser.add_argument('--img_segm_size', dest='img_segm_size', type=int, default=[640,480])#[1280, 720]) #[128,128])#


        train.add_argument('--map_loss_scale', type=float, default=1.0, dest='map_loss_scale')
        train.add_argument('--img_segm_loss_scale', type=float, default=1.0, dest='img_segm_loss_scale')


        train.add_argument('--init_gaussian_weights', dest='init_gaussian_weights', action='store_true',
                            help='initializes the model weights from gaussian distribution')


        train.set_defaults(shuffle_train=True, shuffle_test=True)

        optim = self.parser.add_argument_group('Optim')
        optim.add_argument("--lr_decay", type=float,
                           default=0.99, help="Exponential decay rate")
        optim.add_argument("--wd", type=float,
                           default=0, help="Weight decay weight")

        self.parser.add_argument('--test_iters', type=int, default=20000)

        optimizer_options = self.parser.add_argument_group('Optimizer')
        optimizer_options.add_argument('--lr', type=float, default=0.0002)
        optimizer_options.add_argument('--beta1', type=float, default=0.5)

        #model_options = self.parser.add_argument_group('Model')

        #model_options.add_argument('--with_img_segm', dest='with_img_segm', default=False, action='store_true',
        #                            help='uses the img segmentation pre-trained model during training or testing')
        self.parser.add_argument('--img_segm_model_dir', dest='img_segm_model_dir', default="/home/bo/Desktop/VLN_desktop/VLN_realworld/checkpoints/",
                                    help='job path that contains the pre-trained img segmentation model')


        #self.parser.add_argument('--sem_map_test', dest='sem_map_test', default=False, action='store_true')
        self.parser.add_argument('--offline_eval', dest='offline_eval', default=False, action='store_true')
        self.parser.add_argument('--offline_eval_no_map', dest='offline_eval_no_map', default=False, action='store_true')

        ## Hyperparameters for planning in test navigation


        self.parser.add_argument('--max_steps', type=int, dest='max_steps', default=500,
                                  help='Maximum steps for each test episode')

        self.parser.add_argument('--steps_after_plan', type=int, dest='steps_after_plan', default=20,
                                 help='how many times to use the local policy before selecting long-term-goal and replanning')

        self.parser.add_argument('--stop_dist', type=float, dest='stop_dist', default=2.5,
                                 help='decision to stop distance')
        
        self.parser.add_argument('--success_dist', type=float, dest='success_dist', default=3.0,
                                 help='Radius around the target considered successful')

        self.parser.add_argument('--turn_angle', dest='turn_angle', type=int, default=10,
                                help='angle to rotate left or right in degrees for habitat simulator')
        self.parser.add_argument('--forward_step_size', dest='forward_step_size', type=float, default=0.25,
                                help='distance to move forward in meters for habitat simulator')

        self.parser.add_argument('--save_nav_images', dest='save_nav_images', action='store_true',
                                 help='Keep track and store maps during navigation testing')

        #self.parser.add_argument('--a_1', type=float, dest='a_1', default=0.1,
        #                         help='hyperparameter for choosing long-term-goal')
        #self.parser.add_argument('--a_2', type=float, dest='a_2', default=1.0,
        #                         help='hyperparameter for choosing long-term-goal')

        # options relating to active training (using scenes dataloader)
        self.parser.add_argument('--ensemble_size', type=int, dest='ensemble_size', default=4,
                                help='when using L2M with vln')

        #self.parser.add_argument('--active_training', dest='active_training', default=False, action='store_true')
        #self.parser.add_argument('--img_segm_training', dest='img_segm_training', default=False, action='store_true')

        self.parser.add_argument('--root_path', type=str, dest='root_path', default="/home/bo/Desktop/VLN_desktop/VLN_realworld/")
        self.parser.add_argument('--root_map_dir', type=str, dest='root_map_dir', default="/scratch/bobwu/")

        #self.parser.add_argument('--episodes_root', type=str, dest='episodes_root', default="habitat-api/data/datasets/objectnav/mp3d/v1/")
        self.parser.add_argument('--scenes_dir', type=str, dest='scenes_dir', default='habitat-api/data/scene_datasets/')

        self.parser.add_argument('--stored_episodes_dir', type=str, dest='stored_episodes_dir', default='mp3d_vln_episodes/')
        #self.parser.add_argument('--stored_imgSegm_episodes_dir', type=str, dest='stored_imgSegm_episodes_dir', default='mp3d_objnav_episodes_final_imgSegmOut/')

        #self.parser.add_argument('--active_ep_save_dir', type=str, dest='active_ep_save_dir', default='mp3d_objnav_episodes_active/',
        #                         help='used only during active training to store the episodes')
        #self.parser.add_argument('--max_num_episodes', type=int, dest='max_num_episodes', default=1000,
        #                        help='how many episodes to collect per scene when running the active training')
        self.parser.add_argument('--split', type=str, dest='split', default='train',
                                 choices=['train', 'val', 'val_seen', 'val_unseen', 'test'], help='used only in active training')        

        #self.parser.add_argument('--model_number', type=int, dest='model_number', default=1,
        #                        help='only used when finetuning the model in the active training case - defines which model in ensemble to use')
        #self.parser.add_argument('--finetune', dest='finetune', default=False, action='store_true',
        #                        help='Enable finetuning of an ensemble model')

        #self.parser.add_argument('--uncertainty_type', type=str, dest='uncertainty_type', default='epistemic',
        #                        choices=['epistemic', 'entropy', 'bald'], help='how to estimate uncertainty in active training')

        #self.parser.add_argument('--episode_len', type=int, dest='episode_len', default=10)
        #self.parser.add_argument('--truncate_ep', dest='truncate_ep', default=False,
        #                          help='truncate episode run in dataloader in order to do only the necessary steps, used in store_episodes_parallel')

        #self.parser.add_argument('--occ_from_depth', dest='occ_from_depth', default=True, action='store_true',
        #                        help='if enabled, uses only depth to get the ground-projected egocentric grid')

        self.parser.add_argument('--local_policy_model', type=str, dest='local_policy_model', default='4plus',
                                choices=['2plus', '4plus'])

        self.parser.add_argument('--scenes_list', nargs='+')
        self.parser.add_argument('--gpu_capacity', type=int, dest='gpu_capacity', default=2)

        #self.parser.add_argument('--test_set', type=str, dest='test_set', default='v3', choices=['v3','v5'],
        #                        help='which set of test episodes to use, each has different objects')

        self.parser.add_argument('--occupancy_height_thresh', type=float, dest='occupancy_height_thresh', default=-1.0,
                                help='used when estimating occupancy from depth')

        #self.parser.add_argument('--sem_thresh', dest='sem_thresh', type=float, default=0.75,
        #                        help='used to identify possible targets in the semantic map')

        self.parser.add_argument('--save_img_dir', dest='save_img_dir', type=str, default='test_examples/')

        self.parser.add_argument('--save_test_images', dest='save_test_images', default=False, action='store_true',
                                help='save plots for waypoints and attention during testing')


        ### Added parameters for languange + navigation work

        #self.parser.add_argument('--map_embedding', dest='map_embedding', type=int, default=128,
        #                         help='Map embedding dimensionality')

        self.parser.add_argument('--num_waypoints', dest='num_waypoints', type=int, default=10,
                                 help='Number of waypoints sampled for each trajectory. Affects both sampling and waypoint prediction model')
        self.parser.add_argument('--heatmap_size', dest='heatmap_size', type=int, default=64, # 128
                                 help='Waypoint heatmap size, should match hourglass output size.')

        self.parser.add_argument('--position_loss_scale', type=float, default=1.0, dest='position_loss_scale')
        self.parser.add_argument('--cov_loss_scale', type=float, default=1.0, dest='cov_loss_scale')
        self.parser.add_argument('--heading_loss_scale', type=float, default=1.0, dest='heading_loss_scale')

        self.parser.add_argument('--loss_norm', dest='loss_norm', default=False, action='store_true',
                                 help='If enabled uses the default normalization for L2 loss, and not "sum" ')

        self.parser.add_argument('--eval_set', dest='eval_set', default=False, action='store_true',
                                help='during testing: whether to choose train or eval part of the set the model was trained on')

        ### Attention module hyperparams
        self.parser.add_argument('--d_model', dest='d_model', type=int, default=128,
                                 help='Input embedding dimensionality')
        self.parser.add_argument('--d_ff', dest='d_ff', type=int, default=64, # 256
                                 help='Dimensionality of last feedforward layer')
        self.parser.add_argument('--d_k', dest='d_k', type=int, default=16, # 128, 16
                                 help='Dimensionality of W_k and W_q matrices. This is d_model/n_heads.')
        self.parser.add_argument('--d_v', dest='d_v', type=int, default=16, # 128, 16
                                 help='Dimensionality of W_v matrix. This is d_model/n_heads.')
        self.parser.add_argument('--n_heads', dest='n_heads', type=int, default=8,
                                 help='Number of heads in multi-head attention')
        self.parser.add_argument('--n_layers', dest='n_layers', type=int, default=4,
                                 help='Number of layers in encoder and decoder')

        self.parser.add_argument('--n_hourglass_layers', dest='n_hourglass_layers', type=int, default=2,
                                 help='Number of layers in each hourglass block')

        self.parser.add_argument('--pad_text_feat', dest='pad_text_feat', default=False, action='store_true',
                                 help='Pad text features with zeroes. For batch_size>1 this has to be true')
        
        self.parser.add_argument('--concatenate_inputs', dest='concatenate_inputs', default=False, action='store_true',
                                help='Concatenate map and instruction representation before passing into attention')

        self.parser.add_argument('--token_inputs', dest='token_inputs', type=str, default='bert', choices=['bert', 'shuffle', 'garbage'],
                                help='Bert returns default features for tokens, garbage returns random values')

        self.parser.add_argument('--with_rgb_maps', dest='with_rgb_maps', default=False, action='store_true',
                                help='use both gt labels and rgb maps')

        self.parser.add_argument('--use_first_waypoint', dest='use_first_waypoint', default=False, action='store_true',
                                help='use the first waypoint as an input to the waypoint prediction model')

        self.parser.add_argument('--with_lstm', dest='with_lstm', default=False, action='store_true',
                                help='use lstm layers when predicting the waypoints')


        ### Options related to the actual VLN task
        self.parser.add_argument('--vln', dest='vln', default=False, action='store_true',
                                help='Enables the VLN part of this project')
        self.parser.add_argument('--vln_no_map', dest='vln_no_map', default=False, action='store_true',
                                help='Enables the VLN part of this project were we assume no map is given')
        self.parser.add_argument('--vln_no_map_ext', dest='vln_no_map_ext', default=False, action='store_true',
                                help='Extended model with vln no map case')


        self.parser.add_argument('--num_poses_per_example', dest='num_poses_per_example', type=int, default=10, # 4
                                help='when storing episodes for vln how many poses to use in the same episode')

        self.parser.add_argument('--datasets', nargs='+', default=['R2R_VLNCE_v1-2'], choices=['R2R_VLNCE_v1-2', 'rxr-data'],
                                help='Choose only episodes in vln offline training from the selected datasets')

        self.parser.add_argument('--finetune_bert_last_layer', dest='finetune_bert_last_layer', default=False, action='store_true',
                                help='Finetune only last layer of bert. If disabled, the entire bert is finetuned.')

        self.parser.add_argument('--hourglass_model', dest='hourglass_model', default=False, action='store_true',
                                help='Use hourglass instead of ResNet in the vln case')

        self.parser.add_argument('--map_noise', dest='map_noise', default=False, action='store_true',
                                help='add noise to the input egocentric maps')

        self.parser.add_argument('--goal_conf_thresh', dest='goal_conf_thresh', type=float, default=0.4,
                                help='Goal confidence threshold to decide whether goal is valid')

        self.parser.add_argument('--without_attn_1', dest='without_attn_1', default=False, action='store_true',
                                help='If enabled, does not use the cross-modal attention for map prediction. --vln_no_map should be True')

        self.parser.add_argument('--vln_L2M', dest='vln_L2M', default=False, action='store_true',
                                help='Uses L2M as map predictor. When this is enabled, --vln_no_map should be False')

        self.parser.add_argument('--sample_1', dest='sample_1', default=False, action='store_true', # need for memory issues
                                help='During training, randomly sample a single example from each sequence. If False, it passes the entire sequence.')

        self.parser.add_argument('--consistency_thresh', dest='consistency_thresh', type=float, default=9999,
                                help='Checks if goals are consistently predicted in the same area. A very high value effectively disables this.')

        self.parser.add_argument('--check_floor', dest='check_floor', default=False, action='store_true',
                                help="Check whether the starting position and the goal are on the same floor")

        self.parser.add_argument('--start_episode', dest='start_episode', type=int, default=0,
                                help="Choose which episode index to start the testing")
        self.parser.add_argument('--end_episode', dest='end_episode', type=int, default=0,
                                help="Choose which episode index to end the testing. If 0 then run until length of dataset")