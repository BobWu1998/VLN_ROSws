
import numpy as np
import cv2
import os


def get_frame_seq(im_size, name, ep, video_length, start_ind):

    example_dir = "/home/bo/turtlebot_ws/src/move_cmds/src/semantic_grid/logs/"+name+"/real_world/"
    image_dir = example_dir+"test_examples/ep_"+ep+"/"



    #video_length = 50 # number of frames in the video

    # instruction = cv2.imread(example_dir+"instruction_"+ep+".png")
    # labels = cv2.imread(example_dir+"labels.png")
    #print(instruction.shape)
    print(image_dir+str(0)+"rgb_input.png")

    for i in range(start_ind, video_length):
        rgb = cv2.imread(image_dir+str(i)+"rgb_input.png")
        depth = cv2.imread(image_dir+str(i)+"depth_input.png")
        #print(rgb.shape)
        path_on_pred = cv2.imread(image_dir+str(i)+"_waypoints_on_pred.png")
        ssegm_in = cv2.imread(image_dir+str(i)+"_im_ego_img_segm.png")
        spatial_in = cv2.imread(image_dir+str(i)+"_im_spatial_in.png")
        spatial_pred = cv2.imread(image_dir+str(i)+"_im_spatial_pred.png")
        # path_on_gt = cv2.imread(image_dir+str(i)+"_waypoints_on_gt.png")

        #im_size = (rgb.shape[0], rgb.shape[1])
        rgb = cv2.resize(rgb, im_size, interpolation= cv2.INTER_NEAREST)
        depth = cv2.resize(depth, im_size, interpolation= cv2.INTER_NEAREST)
        path_on_pred = cv2.resize(path_on_pred, im_size, interpolation= cv2.INTER_NEAREST)
        ssegm_in = cv2.resize(ssegm_in, im_size, interpolation= cv2.INTER_NEAREST)
        spatial_in = cv2.resize(spatial_in, im_size, interpolation= cv2.INTER_NEAREST)
        spatial_pred = cv2.resize(spatial_pred, im_size, interpolation= cv2.INTER_NEAREST)
        # path_on_gt = cv2.resize(path_on_gt, im_size, interpolation= cv2.INTER_NEAREST)
        # instruction = cv2.resize(instruction, im_size, interpolation= cv2.INTER_NEAREST)
        # labels = cv2.resize(labels, im_size, interpolation= cv2.INTER_NEAREST)
        #print(path_on_pred.shape)

        #frame_t = np.hstack((instruction, rgb, path_on_pred, path_on_gt, labels))
        # create the top row

        place_holder = np.zeros_like(path_on_pred)
        frame_t_top = np.hstack((rgb, depth, path_on_pred))#, path_on_gt))
        # create bottom row
        frame_t_bottom = np.hstack((ssegm_in, spatial_in, spatial_pred)) #depth #np.hstack((depth, instruction, labels))
        #bot = np.zeros(frame_t.shape)
        #print(bot.shape)
        #print(instruction.shape)
        #bot[:, ]
        frame_t = np.vstack((frame_t_top, frame_t_bottom))

        # frame_t = frame_t_top
        frame_t = np.expand_dims(frame_t, axis=0)
        #print(frame_t.shape)

        if i==start_ind:
            frames = frame_t
        elif i==video_length-1:
            # repeat the last frame multiple times
            frame_t = np.repeat(frame_t, 10, axis=0)
            frames = np.concatenate((frames, frame_t), axis=0)
        else:
            frames = np.concatenate((frames, frame_t), axis=0)
    return frames


im_size = (400,400) #(308,308)
frames = get_frame_seq(im_size=im_size, name="bioe_hallway_example", ep="0", video_length=22, start_ind=0)

print(frames.shape)
fps=5
frame_width = frames.shape[2]
frame_height = frames.shape[1]
# out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
# out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc('H','2','6','4'), fps, (frame_width,frame_height))

out = cv2.VideoWriter('output_video_bioeHallway.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))
for frame in frames:
    out.write(frame)
out.release()


# os.system("ffmpeg -i output_video.mp4 -vcodec libx264 Video.mp4")