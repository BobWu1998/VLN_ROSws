""" Prediction models
"""

#from models.networks import get_network_from_options
from .map_predictor_model import MapPredictorHier


#def get_predictor_from_options(options):
#    segmentation_model = ResNetUNetHierarchical(out1_n_class=options.n_spatial_classes, out2_n_class=options.n_object_classes, without_attn=options.without_attn_1)
#    #return MapPredictorHier(segmentation_model=get_network_from_options(options),
#    return MapPredictorHier(segmentation_model=segmentation_model,
#                        map_loss_scale=options.map_loss_scale,
#                        with_img_segm=True)