#import libraries

import pandas
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

tfd = tfp.distributions
tfpl = tfp.layers

from sklearn.preprocessing import MinMaxScaler
from collections import deque

import os
import sys
import random
import game
import numpy as np
import pickle
 

#conversion from the cell to int and vise versa
cell_to_int_dict = {
    (0, 0): 0,
    (0, 1): 1, 
    (0, 2): 2,
    (1, 0): 3, 
    (1, 1): 4,
    (1, 2): 5,
    (2, 0): 6,
    (2, 1): 7,
    (2, 2): 8
}

# Reversing keys and values
int_to_cell_dict = {v: k for k, v in cell_to_int_dict.items()}


def board_state_int(board, next_player):

    board_state = np.zeros([3, 3, 3])

    for r, row in enumerate(board):
        for c, col in enumerate(row):
            if col == 'X':
                board_state[r, c, 0] = 1
            if col == 'O':
                board_state[r, c, 1] = 2
            if col == ' ':
                board_state[r, c, 2] = next_player

    return board_state

#function for the main training loop
def training_loop(model, memory, epsilon, gamma, save_dir, device, batch_size = 32, current_episode = 0, num_of_episodes = 1000, save_model_every = 20, verbose = True):

    for episode in range(current_episode, num_of_episodes):

        #start game
        ttt = game.TicTacToe()

        #select current player 
        rand_choice = np.random.randint(1, 3, size = 1) #if 1 computer plays first if 2 computer plays second
        if rand_choice == 1:
            computer = 'X'
        else:
            computer = 'O'

        #current state
        cp = 1 if ttt.current_player == 'X' else 2 #since this is the beginning current player is passed on to the first state representation
        state = board_state_int(ttt.board, cp)

        #append current states and moves
        current_game_states, moves = [], []  
        current_game_states.append(state)


        #bool to start game and break the loop
        play_game = True

        while play_game:

            if np.random.rand() <= epsilon:
                all_avail = []
                for row, my_list in enumerate(ttt.board):
                    all_avail.extend((row, index) for index, value in enumerate(my_list) if value == ' ')
                move = random.choice(all_avail)

            else:
                with tf.device(device):
                    q_preds = model.predict(state[np.newaxis, :])
                move_int = np.argmax(q_preds)

                #get the cell from the move integer predicted by the model
                move = int_to_cell_dict[move_int]
        

            if ttt.make_move(move[0], move[1]):
                #store data
                next_player = 2 if ttt.current_player == 'X' else 1
                state = board_state_int(ttt.board, next_player)
                current_game_states.append(state)
                moves.append(move)

                if ttt.check_winner():
                    winner = ttt.winner
                    break
                if all(cell != ' ' for row in ttt.board for cell in row):
                    winner = 'draw'
                    break
                    
                ttt.current_player = 'O' if ttt.current_player == 'X' else 'X'

        #training loop
        if len(memory) > batch_size:
            
            memory_list = list(memory)
            random.shuffle(memory_list)

            #train the model over the current states
            for start_idx in range(0, len(memory_list), batch_size):
                
                end_idx = min(start_idx + batch_size, len(memory_list))
                mini_batch = memory_list[start_idx:end_idx]

                inputs = tf.zeros((end_idx - start_idx, state.shape[0], state.shape[1], state.shape[2]))
                outputs = tf.zeros((end_idx - start_idx, 9)) #9 is the number of values

                #get stuff from the mini batch and get qu values and stuff
                for i, (cs, ns, mv, r) in enumerate(mini_batch):
                    with tf.device(device):
                        q_value = r + gamma * tf.reduce_max(model.predict(ns[np.newaxis, :]))
                        #predicted q values
                        pred_q_values = model.predict(cs[np.newaxis, :])

                    #add the new q value
                    move_int = cell_to_int_dict[mv]
                    pred_q_values[0][move_int] = q_value

                    inputs.numpy()[i] = cs
                    outputs.numpy()[i] = pred_q_values

                model.fit(inputs, outputs, verbose = 0, epochs = 1)
                    
        # assign rewards. If match is drawn either way reward is '0'. If computer choose to play 'X' and winner is 'O' then reward is '-1' vise versa.
        if winner == 'draw':
            reward = 0
        else:
            if winner == computer:
                reward = 1
            else:
                reward = -1

        for i in range(0, len(current_game_states) - 1):
            memory.append((current_game_states[i], current_game_states[i + 1], moves[i], reward))

        #save the model
        if episode % save_model_every == 0:
            
            #save model
            save_path = os.path.join(save_dir, f'model_weight_episode_{episode}.h5')
            model.save_weights(save_path)

            #save memory
            filename = os.path.join(save_dir, f"memory_episode_{episode}.pkl")

            with open(filename, 'wb') as file:
                pickle.dump(memory, file)

        if episode % 10 == 0 and verbose:
            print(f"current episode: {episode}")
        
            