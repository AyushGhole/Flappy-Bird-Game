import flappy_bird_gymnasium
import gymnasium as gym 
from dqn import DQN 
from experience_replay import ReplayMemory 
import itertools
import yaml
import torch 
import torch.nn as nn 
import torch.optim as optim
import random

if torch.backends.mps.is_available(): 
    device = "mps" 
elif torch.cuda.is_available():  
    device = "cuda" 
else: 
    device = "cpu" 


class Agent:
    def __init__(self, params_set): 
        self.param.set = params_set 

        with open("parameters.yaml", "r") as f: 
            all_params_set = yaml.safe.load(f) 
            params = all_params_set[params_set]  
        
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"] 

        self.reward_threshold = params["reward_threshold"] 

        self.loss_fn = nn.MSELoss() 
        self.optimizer = None
        

    def run(self, is_training=True, render=False): 
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None ) 

        num_states = env.observation_space.shape[0] 
        num_actions = env.observation_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)  #input 

    
        if is_training: 
            memory = ReplayMemory(self.replay_memory_size) 
            epsilon = self.epsilon_init     


        for episode in itertools.count(): 

            state, _ = env.reset()  
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_rewards = 0
            terminated = False

            while not terminated: 
                if is_training and random.random() < epsilon: 
                    action = env.action_space.sample()  
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else: 
                    with torch.no_grad():   
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                next_state, reward, terminated, _, _ = env.step(action.item()) 

                #create our tensors 
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)


                if is_training: 
                    memory.append((state, action,  next_state, reward, terminated)) 

                    state = next_state
                    episode_rewards += reward

            print(f"for episode={episode+1} with total rewards={episode_rewards} & epsilon={epsilon}") 

            #Epsilon Deccay code 
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min) 
             
            # env.close() 