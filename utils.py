from tensorforce import Agent


def create_agent(environment):
    agent = Agent.create(
            agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3, saver=dict(directory='model', frequency=10, max_checkpoints=5)
            # agent='ac', environment=environment, batch_size=10, memory=11000, learning_rate=1e-3, saver=dict(directory='model', frequency=10, max_checkpoints=5)
        )
    # agent = Agent.create(
    #         agent='vpg', environment=environment, batch_size=10, learning_rate=1e-3, saver=dict(directory='model',frequency=10,max_checkpoints=5)
    #     )
    # agent = Agent.create(
    #         agent='ac', environment=environment, batch_size=10, memory=11000, learning_rate=1e-3, saver=dict(directory='model', frequency=10, max_checkpoints=5)
    #     )
    return agent


def run(environment, agent, n_episodes, max_step, test=False):
    # environment.CompilerModel.max_step = max_step
    # Loop over episodes
    # Initialize episode
    episode_length = 0
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:         # 这里是轮次循环的开始
        # Run episode
        episode_length += 1
        actions = agent.act(states=states)   # agent与环境environment进行交互，每一次交互中智能体观察到环境的state
        states, terminal, reward = environment.execute(actions=actions)     # 基于这个状态采取一个动作（action）
        agent.observe(terminal=terminal, reward=reward)   # 环境会根据这个动作返回一个奖励（reward）和下一个状态（next state）


def runner(
    environment,
    agent,
    max_step_per_episode,
):

    for i in range(1):  # Divide the number of episodes into batches of 100 episodes
        # Train Agent for 100 episode
        run(environment, agent, 1, max_step_per_episode)
        print("The trainning is over")
        print(environment.rewardlist)
        print(environment.steplist)
        print(environment.ncdlist)
    agent.close()
    environment.close()