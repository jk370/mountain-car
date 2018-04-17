# Please write your code in this cell. You can add additional code cells below this one, if you would like to.
import numpy as np
import mountaincar
import matplotlib.pyplot as plt
%matplotlib inline

def make_tiles(position_min=-1.2, position_max=0.5, velocity_min =-0.07, velocity_max=0.07, tile_number=9, overlap=10, actions=3):
    '''Makes tiles from given position and velocity boundaries'''
    # Set uniform tile widths (for overlap)
    position_tile_width = (position_max - position_min) / tile_number
    velocity_tile_width = (velocity_max - velocity_min) / tile_number
    
    # Set initial tile locations - 10x10 tiles to allow for offset
    initial_position_tile = np.linspace(position_min, position_max+position_tile_width, tile_number+1)
    initial_velocity_tile = np.linspace(velocity_min, velocity_max+velocity_tile_width, tile_number+1)
    
    # Initialize tiling array
    all_tilings = []
    
    # Make overlapping tiles for each possible action
    for _ in range(actions):
        # Set up initial tile arrays
        position_tiles = [np.copy(initial_position_tile)]
        velocity_tiles = [np.copy(initial_velocity_tile)]
    
        # Loop through and overlap 9 more tiles
        for _ in range(overlap-1):
            position_offset = np.random.uniform(0, position_tile_width)
            velocity_offset = np.random.uniform(0, velocity_tile_width)
            position_tiles.append(initial_position_tile - position_offset)
            velocity_tiles.append(initial_velocity_tile - velocity_offset)
            
        all_tilings.append((position_tiles, velocity_tiles))
        
    return all_tilings
    
def find_features(tiles, state, action):
    '''Returns the tile corresponding to the car position and velocity for each tiling'''
    # Initialize required variables
    index = action+1
    position_tiles, velocity_tiles = tiles[index]
    car_position, car_velocity = state
    features = []
    
    # Loop through tiles and digitize to find features
    for i in range(len(position_tiles)):
        position_tile = (np.digitize(car_position, position_tiles[i])).item()
        velocity_tile = (np.digitize(car_velocity, velocity_tiles[i])).item()
        features.append((position_tile, velocity_tile))
    
    return features

def evaluate_theta(theta_table, features, action, action_number=3):
    '''Returns theta sum for on features - only works if tilings are same size in both dimensions'''
    tiles = len(theta_table)
    overlap = len(features)
    tiling_number = int((tiles/overlap/action_number))
    dimension = int(np.sqrt(tiling_number))
    theta_sum = 0
    
    # Loop through each feature and find corresponding theta value
    for i in range(len(features)):
        pos_index, vel_index = features[i]
        table_index = np.ravel_multi_index([pos_index, vel_index], (dimension, dimension))
        table_index += (tiling_number*i) + ((action+1)*(tiling_number*overlap))
        theta_sum += theta_table[table_index]
    
    return theta_sum

def learn(alpha=0.15, epsilon=0, gamma=1, lamb=0.9, episodes=100, trace="replace"):
    '''Performs linear, gradient-descent Sarsa learning'''
    # Perform initializations
    episode_rewards = []
    env = mountaincar.MountainCar()
    tiles = make_tiles()
    action_size = len(tiles)
    tile_overlap = len(tiles[0][0])
    tile_size = len(tiles[0][0][0])*len(tiles[0][1][0])
    tile_number = action_size*tile_size*tile_overlap
    theta = np.zeros(tile_number)
    
    # Run number of episodes for learning
    for _ in range(episodes):
        env.reset()
        total_reward = 0
        eligibility = np.zeros(tile_number)
        env_state = (env.position, env.velocity)
        chosen_action = np.random.choice(env.actions)
        features = find_features(tiles, env_state, chosen_action)
        
        # Repeat for each step in episode
        while not env.game_over:
            # Action the appropriate trace
            for i in range(len(features)):
                # Convert 3D state-action tile features to 1D index for theta table
                pos_index, vel_index = features[i]
                index = np.ravel_multi_index([pos_index, vel_index], (len(tiles[0][0][0]), len(tiles[0][1][0])))
                index += (tile_size*i) + ((chosen_action+1)*(tile_size*tile_overlap))
                if trace == "accumulate":
                    eligibility[index] += 1
                elif trace == "replace":
                    eligibility[index] = 1
                else:
                    print("Unknown trace type")
                    break
                    
            # Take action and observe next state and reward
            reward = env.make_step(action=chosen_action)
            env_state = (env.position, env.velocity)
            delta = reward - evaluate_theta(theta, features, chosen_action)
            
            # Perform epsilon-greedy action
            chance = np.random.uniform(0,1)
            if (chance >= epsilon):
                Q_actions = []
                for action in env.actions:
                    features = find_features(tiles, env_state, action)
                    Q_actions.append(evaluate_theta(theta, features, action))
                # Find choose maximum action
                chosen_action = env.actions[np.argmax(Q_actions)]
                features = find_features(tiles, env_state, chosen_action)
                Q_action = evaluate_theta(theta, features, chosen_action)

            else:
                chosen_action = np.random.choice(env.actions)
                features = find_features(tiles, env_state, chosen_action)
                Q_action = evaluate_theta(theta, features, chosen_action)
            
            # Make step updates
            delta += (gamma * Q_action)
            theta += (alpha * delta * eligibility)
            eligibility *= (gamma*lamb)
            total_reward += reward
            #env.plot()
        
        # Add total reward for episode to array (to allow plotting) and return
        episode_rewards.append(total_reward)
    
    return episode_rewards

def plot_average_curve(agents=50):
    '''Plots average learning curve for a number of agents'''        
    plt.figure(1)
    average_rewards = []
    
    for _ in range(agents):
        learning_rewards = learn(episodes = 100)
        average_rewards.append(learning_rewards)
        
    average_rewards = np.mean(average_rewards, axis = 0)
    
    plt.plot(average_rewards)
    plt.title("Average learning rate for 50 agents - LGD Sarsa Lambda")
    plt.xlabel("Episode number")
    plt.ylabel("Episode Reward")
    
def plot_single_curve():
    '''Plots learning curve for a single agent'''
    plt.figure(2)
    learning_rewards = learn(episodes = 100)
    plt.plot(learning_rewards)
    plt.title("Learning for single agent - LGD Sarsa Lambda")
    plt.xlabel("Episode number")
    plt.ylabel("Episode Reward")
    
#plot_average_curve()
#plot_single_curve()