

#from .resnetUnetHierarchical import ResNetUNetHierarchical
from .map_encoder import MapEncoder
from .transformer import MapAttention
from .hourglass import StackedHourglass
from .hourglass_goal import StackedHourglassGoal
from .resnetUnetGoalPred import ResNetUNetGoalPred
from .resnetUnetHierarchical import ResNetUNetHierarchical
from .resnetUnet import ResNetUNetAttn

'''
Model ResNetUnet taken from:
https://github.com/usuyama/pytorch-unet
'''

#def get_network_from_options(options):
#    """ Gets the network given the options
#    """
#    return ResNetUNetHierarchical(out1_n_class=options.n_spatial_classes, out2_n_class=options.n_object_classes, with_img_segm=options.with_img_segm)