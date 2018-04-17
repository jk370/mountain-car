# Please write your code in this cell. You can add additional code cells below this one, if you would like to.
import numpy as np
import mountaincar
import lgdsarsa
import matplotlib.pyplot as plt
%matplotlib inline

def watkins_q(alpha=0.15, epsilon=0, gamma=1, lamb=0.9, episodes=100, trace="replace"):
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
            
            # Update delta from max Qa
            Q_actions = []
            for action in env.actions:
                features = find_features(tiles, env_state, action)
                Q_actions.append(evaluate_theta(theta, features, action))
            
            delta += (gamma * max(Q_actions))
            theta += (alpha * delta * eligibility)
            
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
                eligibility *= (gamma*lamb)

            else:
                chosen_action = np.random.choice(env.actions)
                eligibility *= 0
            
            total_reward += reward
            #env.plot()
        
        # Add total reward for episode to array (to allow plotting) and return
        episode_rewards.append(total_reward)
    
    return episode_rewards

def watkins_plot_average_curve(agents=50):
    '''Plots average learning curve for a number of agents'''        
    plt.figure(3)
    average_watkins_rewards = []
    average_watkins_epsilon_rewards = []
    average_sarsa_rewards = []
    
    for i in range(agents):
        sarsa_learning_rewards = learn(episodes = 100)
        watkins_learning_rewards = watkins_q(episodes = 100)
        watkins_epsilon_rewards = watkins_q(epsilon = 0.1, episodes = 100)
        
        average_sarsa_rewards.append(sarsa_learning_rewards)
        average_watkins_rewards.append(watkins_learning_rewards)
        average_watkins_epsilon_rewards.append(watkins_epsilon_rewards)
        print(i)
        
    average_sarsa_rewards = np.mean(average_sarsa_rewards, axis = 0)
    average_watkins_rewards = np.mean(average_watkins_rewards, axis = 0)
    average_watkins_epsilon_rewards = np.mean(average_watkins_epsilon_rewards, axis = 0)
    
    plt.plot(average_sarsa_rewards)
    plt.plot(average_watkins_rewards)
    plt.plot(average_watkins_epsilon_rewards)
    plt.title("Average learning across 50 agents for Sarsa-lambda versus Watkins-Q (with and without epsilon)")
    plt.xlabel("Episode number")
    plt.ylabel("Episode Reward")
    plt.legend(["Sarsa-lambda","Watkins-Q", "Watkins-Q (e=0.1)"], loc = 'lower right')
    
def watkins_plot_single_curve():
    '''Plots learning curve for a single agent'''
    plt.figure(4)
    learning_rewards = watkins_q(episodes = 100)
    plt.plot(learning_rewards)
    plt.title("Learning for single agent - Watkins-Q")
    plt.xlabel("Episode number")
    plt.ylabel("Episode Reward")
    
#watkins_plot_average_curve(agents = 50)
#watkins_plot_single_curve()