import numpy as np
import skimage
import json

from collections import deque


def get_config(config_file_path):
    """
    Reads configuration variables form file.
    """

    return json.load(open(config_file_path, 'r'))

def preproc_frame(frame, frame_size):
    """ 
    This function converts a single frame to grayscale and crops useful parts
    returns 2D array.
    """

    return skimage.transform.resize(
        ((frame.mean(axis=-1,keepdims=1)).squeeze()[30:-10,10:-10])/255.0,
        [frame_size, frame_size], 
        anti_aliasing=True,
        mode='reflect'
    )


def stack_frames(stacked_frames, raw_frame, new_ep, stack_size):
    """
    Stack multiple frames together in a queue.

    If it is a new episode stack multiple times the same image to fill the
    queue. 
    """
    
    frame = preproc_frame(raw_frame)
    
    if new_ep:
        # Because we're in a new episode, copy the same frame 4x
        for i in range(stack_size): stack_frames.append(frame) 
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

    # Stack the frames
    stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames
