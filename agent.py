import flappy_bird_gymnasium
import gymnasium as gym 
from dqn import DQN 
from experience_replay import ReplayMemory

if torch.backends.mps.is_available(): 
    device = "mps" 
elif torch.cuda.is_available():  
    device = "cuda" 
else: 
    device = "cpu" 

env = gym.make("FlappyBird-v0", render_mode="human")

state, _ = env.reset() 

def run(self, is_training=True, render=False):
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None ) 

    num_states = env.observation_space.shape[0] 
    num_actions = env.observation_space.n
    
    policy_dqn = DQN(num_states, num_actions).to(device)  #input 

    state, _ = env.reset() 

    while True:
        # Next action:
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        # Processing: terminated => Done
        next_state, reward, terminated, _, _ = env.step(action)
    
        # Checking if the player is still alive
        if terminated:
            break

    env.close() 