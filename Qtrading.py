u"""
Neural Network based trading algorithm 
trained with reinforcement learning

Adapted from: https://keon.io/deep-q-learning/

Created on 23/03/16
by fccoelho
license: GPL V3 or Later
"""


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, Nadam
from keras.callbacks import TensorBoard
from queue import deque
import numpy as np
import random
import pandas as pd
from poloniex import get_ohlc


class DQNAgent:
    """
    Q-Learning agent
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.05  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Input layer, this must encode the current state of the system (a time window to the past(prices, volume, asks and bids)
        # and whether you have a position in the market, volume and price.

        model = Sequential()
        model.add(Dense(164, init='lecun_uniform', input_dim=self.state_size, activation='relu'))
        #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        model.add(Dense(150, init='lecun_uniform', activation='relu'))
        #model.add(Dropout(0.2))

        # 3 output units, for three possible actions: Buy, Sell, Pass
        model.add(Dense(3, init='lecun_uniform', input_dim=self.action_size, activation='softmax'))
        # Softmax activation returns value between 0 and 1 which can be interpreted as probability of each action

        # rms = RMSprop()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                                  np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[TB_callback])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Environment:
    def __init__(self, cash, port):
        self.cash = [cash]
        self.port = [port]
        self.osize = 0.05

    def step(self, action, state):
        actions = np.array(['buy', 'sell', 'pass'])
        a = actions[np.argmax(action)]
        if a == 'buy':
            new_cash = self.cash[-1] - self.osize * self.cash[-1]
            new_port = self.port[-1] + (self.osize * self.cash[-1]) / state[0]
        elif a == 'sell':
            new_cash = self.cash[-1] + self.port[-1]*self.osize * state[0]
            new_port = self.port[-1] - self.osize * self.port[-1]
        else:
            pass
        reward = (new_cash-self.cash[-1])+(new_port-self.port[-1]*state[0])
        if new_cash <= 0:
            done = True
        self.cash.append(new_cash)
        self.port.append(new_port)
        return np.array([state[0], self.port[-1], self.cash[-1]]), reward, False

TB_callback = TensorBoard(log_dir='./tensorboard',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True,
                              # embeddings_freq=10
                              )

if __name__ == "__main__":
    episodes = 200
    import datetime
    # Obtain the data
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=364)
    data = get_ohlc("USDT_BTC", start, end)

    # state = [position(T/F), how much, price, market price, volume]
    agent = DQNAgent(3, 3)
    initial_cash = 10000

    # Iterate the game
    for e in range(episodes):
        env = Environment(initial_cash, 0)
        # reset state in the beginning of each episode
        print('Episode: {}'.format(e))
        total = 0
        cash = initial_cash
        state = np.array((data.iloc[0]['close'], total, cash))

        # time_t represents each trade in history
        for time_t in range(len(data)):

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done = env.step(action, state)
            next_state = np.reshape(next_state, [1, 3])
            print('Reward: {}'.format(reward))

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break
        # train the agent with the experience of the episode
        agent.replay(32)

