import gymnasium as gym


def play_blackjack(agent):
    env1 = gym.make("Blackjack-v1", sab=True)
    env2 = gym.make("Blackjack-v1", sab=True)

    score1 = 0
    score2 = 0
    play_game = True
    while play_game:
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        done1 = False
        done2 = False

        print("\nnew game")
        # player 1
        done1, obs1, agent, env1 = get_agent_action(done1, obs1, agent, env1)

        # player 2
        done2, obs2, env2 = get_player_action(done2, obs2, env2)

        score1, score2 = calculate_scores(obs1, obs2, score1, score2)

        play_game = play_again(score1, score2)

    env1.close()
    env2.close()


def get_agent_action(done, obs, agent, env):
    print("the ai agent is playing")
    while not done:
        action = agent.get_action(obs)
        if action == 1:
            print("the agent decided to hit")
        else:
            print("the agent decided to stand")
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return done, obs, agent, env


def get_player_action(done, obs, env):
    print("it is your turn to play")
    while not done:
        print(f"your current hand: {obs[0]}")
        while True:
            action = input("press 0 to stand, or press 1 to hit")
            if action in ["0", "1"]:
                action = int(action)
                break
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return done, obs, env


def calculate_scores(obs1, obs2, score1, score2):
    print(f"ai score: {obs1[0]} \nyour score: {obs2[0]}")
    if obs1[0] > 21 & obs2[0] > 21:
        pass
    elif obs1[0] > 21:
        print("ai busted; you won")
        score2 += 2
    elif obs2[0] > 21:
        print("you busted, ai won")
        score1 += 2
    elif obs1[0] > obs2[0]:
        print("ai won")
        score1 += 2
    elif obs1[0] < obs2[0]:
        print("you won")
        score2 += 2
    else:
        print("tie")
        score1 += 1
        score2 += 1
    return score1, score2


def play_again(score1, score2):
    print(f"overall score: \nai score: {score1} \nyour score: {score2}")
    while True:
        again = input("Pres y play again or any other key to quit")
        return True if again in ["y", "Y"] else False
