import matplotlib.pyplot as plt
import numpy as np
import skimage
import json
import random
import time

from collections import deque
from vizdoom import *


def create_environment(config, visible=True):
    """
    Function to define a new game environment.
    """

    game = DoomGame()
    
    # Load the correct configuration
    game.load_config(config['game_config'])
    
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path(config['game_scenario'])
    
    game.set_window_visible(visible)

    game.init()
    
    actions = [config['actions'][key] for key in config['actions']]

    return game, actions


def test_environment(config):
    """
    Function to test the basic game environment.
    """

    game, actions = create_environment(config)

    episodes = 10

    for i in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables

            action = random.choice(actions)
            print(action)

            reward = game.make_action(action)
            print ("\treward:", reward)

            time.sleep(0.02)

        print ("Result:", game.get_total_reward())
        time.sleep(1)
        
    game.close()


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

    frame = np.transpose(frame, (1, 2, 0))

    return skimage.transform.resize(
        ((frame.mean(axis=-1,keepdims=1)).squeeze()[60:-30,40:-40])/255.0,
        [frame_size, frame_size],
        mode='reflect'
    )


def stack_frames(stacked_frames, raw_frame, new_ep, stack_size, frame_size):
    """
    Stack multiple frames together in a queue.

    If it is a new episode stack multiple times the same image to fill the
    queue. 
    """
    
    frame = preproc_frame(raw_frame, frame_size)
    
    if new_ep:
        # Because we're in a new episode, copy the same frame 4x
        for i in range(stack_size): stacked_frames.append(frame) 
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

    # Stack the frames
    stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state


class Memory():
    """
    Class for implementing Experience Replay.
    """

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)

        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False
        )
        
        return [self.buffer[i] for i in index]


def pretrain(pretrain_steps, memory, stack_size, frame_size, stacked_frames, game, actions):
    """
    Perform the pre-training filling the memory buffer.
    """

    game.new_episode()

    for i in range(pretrain_steps):
        
        if i == 0:
            frame = game.get_state().screen_buffer
            state = stack_frames(
                stacked_frames, frame, True, stack_size, frame_size
            )

        action_index = np.random.randint(len(actions))
        action = actions[action_index]

        reward = game.make_action(action)
        done = game.is_episode_finished()

        if done:
            next_state = np.zeros(state.shape)
            memory.add((state, action_index, reward, next_state, done))
            
            game.new_episode()
            
            frame = game.get_state().screen_buffer
            state = stack_frames(
                stacked_frames, frame, True, stack_size, frame_size
            )

        else:
            next_frame = game.get_state().screen_buffer
            next_state = stack_frames(
                stacked_frames, next_frame, False, stack_size, frame_size
            )
            
            memory.add((state, action_index, reward, next_state, done))
            state = next_state


class LinearSchedule(object):      

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):                                                                                                                                                                                                     
        '''
        Linear interpolation between initial_p and final_p over                                                                                                                                                                                                      
        schedule_timesteps. After this many timesteps pass final_p is                                                                                                                                                                                                   
        returned.                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                        
        Args:                                                                                                                                                                                                                                                    
            - schedule_timesteps: Number of timesteps for which to linearly 
                anneal initial_p to final_p                                                                                                                                                                                                                                                  
            - initial_p: initial output value                                                                                                                                                                                                                                        
            -final_p: final output value                                                                                                                                                                                                                                          
        '''

        self.schedule_timesteps = schedule_timesteps                                                                                                                                                                                                                    
        self.final_p = final_p                                                                                                                                                                                                                                          
        self.initial_p = initial_p                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                         
    def value(self, t):    
        """
        Will return the value of epsilon at time t.
        """

        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def plot_reward_curve(rewards, title, smooth):
    n_episodes = len(rewards)
    avgs = []
    smoothed = []
    
    avg = 0.0
    for ep in range(1, n_episodes+1):
        avg += rewards[ep - 1]
        
        if ep % smooth == 0:
            smoothed.append(avg / float(smooth))
            avg = 0.0
    
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    ax.set(title=title, xlabel="Episodes", ylabel="Average Reward")    
    ax.plot(range(len(smoothed)), smoothed, 'b')
    plt.show()
