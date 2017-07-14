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
from keras.optimizers import RMSprop, Adam
from queue import deque
import numpy as np
import random
import pandas as pd


def get_price_history(pair, start, end):
    """
    Poloniex API only returns maximum of 300000 trades or 1 year for each pair.
    :returns:
    dictionary with one dataframe per pair
    """
    print('Downloading {} from {} to {}.'.format(pair, start, end))
    url = 'https://poloniex.com/public?command=returnTradeHistory&currencyPair={}&start={}&end={}'.format(pair,
                                                                                                          int(start.timestamp()),
                                                                                                          int(end.timestamp()))
    df = pd.read_json(url)
    df.set_index(['date'], inplace=True)
    print('fetched {} {} trades.'.format(df.size, pair))
    df = df.resample('1T').mean()  # resample in windows of 1 minute

    return df

def get_ohlc(pair, start, end):
    print('Downloading {} from {} to {}.'.format(pair, start, end))
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period=300'.format(pair,
                                                                                                          int(start.timestamp()),
                                                                                                          int(end.timestamp()))
    df = pd.read_json(url)
    df.set_index(['date'], inplace=True)
    return df




class DQNAgent:
    """
    Q-Learning agent
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
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

        model.add(Dense(150, init='lecun_uniform', activation=relu))
        #model.add(Dropout(0.2))

        # 3 output units, for three possible actions: Buy, Sell, Pass
        model.add(Dense(3, init='lecun_uniform', input_dim=self.action_size, activation='softmax'))
        # Softmax activation returns value between 0 and 1 which can be interpreted as probability of each action

        # rms = RMSprop()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

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
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    episodes = 200
    import datetime
    # Obtain the data
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=364)
    data = get_price_history("USDT_BTC", start, end)
    # state = [position(T/F), how much, price, market price, volume]
    agent = DQNAgent(5, 3)

    # Iterate the game
    for e in range(episodes):

        # reset state in the beginning of each episode
        # TODO: add balances here
        state = np.array((data.iloc[0]['rate'], data.iloc[0]['total']))

        # time_t represents each trade in history
        for time_t in range(500):

            # Decide action
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

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

