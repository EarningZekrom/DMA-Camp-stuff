
import gym
import gym.utils.play
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
import numpy as np
import gym
import random

#env = gym.make("FishingDerby-v0")

#from gym.utils import play


def initial_data(number_of_games, game_turns, acceptable_score, classes, len_data):
    x = []
    y = []

    l_1hot = [0 for i in range(classes)]

    for i in range(number_of_games):
        env.reset()

        game_memory = []

        prev_observation = []

        score = 0

        for turn in range(game_turns):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score += int(reward)

            if turn > 0:
                game_memory.append([prev_observation, int(action)])

            prev_observation = observation

            if done == True:
                break

        if score >= acceptable_score:
            for data in game_memory:
                x.append(data[0])
                # x.append(np.array(data[0]).reshape(1, len_data))
                label = list(l_1hot)
                label[data[1]] = 1
                y.append(np.array(label))

    print('{} examples were made.'.format(len(x)))
    x = np.array(x)
    y = np.array(y)
    return x, y
    # return np.array(x).reshape(-1, 1, len_data), np.array(y).reshape(-1, 1, classes)


def play_game(n_games, n_moves, model=None):
    for i in range(n_games):
        prev_observation = env.reset()
        score = 0

        for step in range(n_moves):
            action = np.argmax(model.predict(np.array([prev_observation])))
            observation, reward, done, info = env.step(action)
            env.render()
            prev_observation = observation
            score += int(reward)
            state = observation
            if done == True:
                break

env = gym.make("Enduro-v0")

from gym.utils import play

gym.utils.play.play(env, zoom = 4)







action_space = env.action_space.n
observation_space = env.observation_space.shape

print (action_space)

model = Sequential()
model.add(Dense(100, input_shape=observation_space))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(action_space, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy")

X, Y = initial_data(100000, 200, 50, action_space, observation_space[0])

print(X.shape)
print(Y.shape)

model.fit(X, Y,
          batch_size=512, epochs=1)

play_game(10, 5000, model)