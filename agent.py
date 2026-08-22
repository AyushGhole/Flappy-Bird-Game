import flappy_bird_gymnasium
import gymnasium as gym 
from dqn import DQN 
from experience_replay import ReplayMemory 
import itertools
import yaml
import torch 
import torch.nn as nn 
import torch.optim as optim


if torch.backends.mps.is_available(): 
    device = "mps" 
elif torch.cuda.is_available():  
    device = "cuda" 
else: 
    device = "cpu" 


class Agent: 
    def run(self, is_training=True, render=False): 
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None ) 

        num_states = env.observation_space.shape[0] 
        num_actions = env.observation_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)  #input 

    
        if is_training: 
            memory = ReplayMemory(10000)     


        for episode in itertools.count(): 

            state, _ = env.reset() 
            episode_rewards = 0
            terminated = False

            while not terminated:
                action = env.action_space.sample()

                next_state, reward, terminated, _, _ = env.step(action)

                if is_training: 
                    memory.append((state, action,  next_state, reward, terminated)) 

                    state = new_state
                    episode_rewards += reward

            print(f"for episode={episode+1} with total rewards={episode_rewards}")
            # env.close() 