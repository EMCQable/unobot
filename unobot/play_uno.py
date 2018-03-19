import gym
import gym_uno

from baselines import deepq


def main():
    env = gym.make("Uno-v0")
    act = deepq.load("uno_model.pkl")
    wins = 0
    games = 0

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act([obs[0:23]])[0])
            episode_rew += rew
        games += 1
        if (episode_rew > 100):
            wins += 1
        winpercent = round(wins/games, 3)
        print("Episode reward", episode_rew,"\t Win %: ", winpercent, "\t\tGames:", games)


if __name__ == '__main__':
    main()
