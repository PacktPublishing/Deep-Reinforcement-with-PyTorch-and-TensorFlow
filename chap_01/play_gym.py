import gym.spaces
# Choose environment
# env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('LunarLander-v2')
env = gym.make('BipedalWalker-v2')
# env = gym.make('CarRacing-v0')
# env = gym.make('Pong-v0')
# Show action and state space
print(env.action_space)
print(env.observation_space)

for i_episode in range(5):
    print('Episode:', i_episode)
    done = False
    t = 0
    # Reset environment
    state = env.reset()
    # For t timesteps
    while not done:
        # Display results on screen
        env.render()
        # Take random action
        action = env.action_space.sample()
        # Send action to environment and get next state, reward, terminal, and debug_info
        state, reward, done, info = env.step(action)
        # Episode end
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        t += 1

# Close environment
env.close()
