""" Entry point for training
"""
from train_options import TrainOptions
# from trainer_waypoints import Trainer
# from trainer_vln import TrainerVLN
# from trainer_vln_no_map import TrainerVLN_UnknownMap
# from trainer_vln_L2M import TrainerVLN_L2M
# from tester_waypoints import WaypointTester
# from tester import VLNTester, WaypointVLNTester
from test_external import VLNTesterUnknownMap

import multiprocessing as mp
from multiprocessing import Pool, TimeoutError




if __name__ == '__main__':
    options = TrainOptions().parse_args()
    tester = VLNTesterUnknownMap(options)
    tester.pipeline_integration()

