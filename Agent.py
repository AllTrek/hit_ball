import keras
from collections import deque
import numpy as np
import random



class DQNAgent():

    def __init__(self, replay_memory_size, action_space_size, min_replay_memory_size, minibatch_size,
                 update_target_every, discount):

        self.REPLAY_MEMORY_SIZE = replay_memory_size
        self.ACTION_SPACE_SIZE = action_space_size
        self.MIN_REPLAY_MEMORY_SIZE = min_replay_memory_size
        self.MINIBATCH_SIZE = minibatch_size
        self.UPDATE_TARGET_EVERY = update_target_every
        self.DISCOUNT = discount

        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0


    def create_model(self):

        model = keras.Sequential()
        # model = keras.models.load_model("")

        model.add(keras.layers.Dense(input_shape=(12,), units=256))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Dense(units=128))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Dense(units=128))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Dense(units=128))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Dense(units=self.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        # print("Transition: ")
        # print(transition)
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, 12))[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        current_states = ([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(np.array(current_states).reshape(-1, 12)) # This should produce 3 results

        new_current_states = [transition[3] for transition in minibatch]
        future_qs_list = self.target_model.predict(np.array(new_current_states).reshape(-1, 12))

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q


            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False)
        print("Model.fit called")

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
