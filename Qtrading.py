u"""
Neural Network based trading algorithm 
trained with reinforcement learning

Created on 23/03/16
by fccoelho
license: GPL V3 or Later
"""


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop


# Input layer, this must encode the current state of the system (a time window to the past(prices, volume, asks and bids)
# and whether you have a position in the market, volume and price.

model = Sequential()
model.add(Dense(164, init='lecun_uniform', input_shape=(64,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

# 3 output units, for three possible actions: Buy, Sell, Pass
model.add(Dense(3, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

#Todo: implement main loop
#TODO: implement function to calculate reward.
