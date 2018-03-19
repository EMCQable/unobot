
import gym
import gym_uno
from baselines import deepq


def main():
    env = gym.make("Uno-v0")
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([64], layer_norm=True)
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=500000,
        buffer_size=50000,
        exploration_fraction=0.3,
        exploration_final_eps=0.1,
        print_freq=5,
        param_noise=True
    )
    print("Saving model to uno_model.pkl")
    act.save("uno_model.pkl")


if __name__ == '__main__':
    main()
