
#import necessary packages
import numpy as np
import skimage
from collections import deque



#This function converts a single frame to grayscale and crops useful parts
def preproc_frame(frame):
    return skimage.transform.resize(
        ((frame.mean(axis=-1,keepdims=1)).squeeze()[30:-10,10:-10])/255.0,
        [128, 128], 
        anti_aliasing=True,
        mode='reflect'
    )


#Pass Stack size as parameter
# stack_size = 4 
def stack_frames(stacked_frames, raw_frame, new_ep):
    ##Input Args:
        # stacked_frames: current deque of frames
        # raw_frame: the color/unedited frame that we pass to be preprocessed
        # new_ep: boolean indicates whether we should make a new stack frame
    
    # Preprocess frame
    frame = preproc_frame(raw_frame)
    
    if new_ep:
        # Clear existing stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames
