# import numpy as np
# import datasets.util.viz_utils as viz_utils

# file = open('/home/bo/Desktop/VLN_all/data/camera_csv/_Depth_1655240137885.53833007812500.csv', 'rb')
# depth_abs = np.loadtxt(file,delimiter = ",")
# print('log-----')

# print("depth_abs size",depth_abs.shape)
# # depth_abs = torch.tensor(depth_abs[0:self.test_ds.img_size[0], 0:self.test_ds.img_size[1]], device='cuda')

# # imageSize = (640, 360, 1)#(self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
# # # crop the depth image

# imageSize = (720, 720, 1)
# # crop depth_abs to img_size
# depth_abs = depth_abs[0:imageSize[0], 300:1020].reshape(imageSize)

# viz_utils.vis_arr(depth_abs, './', 'test')

import matplotlib.pyplot as plt
import numpy as np
num_cls = {
    0: 'void',
    1: 'chair',
    2: 'door',
    3: 'table',
    4: 'cushion',
    5: 'sofa',
    6: 'bed',
    7: 'plant',
    8: 'sink',
    9: 'toilet',
    10: 'tv_monitor',
    11: 'shower',
    12: 'bathtub',
    13: 'counter',
    14: 'appliance',
    15: 'structure',
    16: 'other',
    17: 'free-space',
    18: 'picture',
    19: 'cabinet',
    20: 'chest_of_drawers',
    21: 'stool',
    22: 'towel',
    23: 'fireplace',
    24: 'gym-equipment',
    25: 'sesating',
    26: 'clothes'
}
color_mapping_27 = {
    0:(255,255,255), # white
    1:(128,128,0), # olive (dark yellow)
    2:(0,0,255), # blue
    3:(255,0,0), # red
    4:(255,0,255), # magenta
    5:(0,255,255), # cyan
    6:(255,165,0), # orange
    7:(255,255,0), # yellow
    8:(128,128,128), # gray
    9:(128,0,0), # maroon
    10:(255,20,147), # pink 
    11:(0,128,0), # dark green
    12:(128,0,128), # purple
    13:(0,128,128), # teal
    14:(0,0,128), # navy (dark blue)
    15:(210,105,30), # chocolate
    16:(188,143,143), # rosy brown
    17:(0,255,0), # green
    18:(255,215,0), # gold
    19:(0,0,0), # black
    20:(192,192,192), # silver
    21:(138,43,226), # blue violet
    22:(255,127,80), # coral
    23:(238,130,238), # violet
    24:(245,245,220), # beige
    25:(139,69,19), # saddle brown
    26:(64,224,208) # turquoise
}
rgb_array = np.zeros((27,1,3))
names = []

for i in range(27):
    for channel in range(3):
        rgb_array[i, 0, channel] = color_mapping_27[i][channel]
        names.append(num_cls[i])

img = np.array(rgb_array, dtype=int)
plt.imshow(img)#, extent=[0, 16000, 0, 1], aspect='auto')


for i in range(27):
    plt.text(0.55, 0.5+i, num_cls[i])
plt.xticks([])
plt.show()