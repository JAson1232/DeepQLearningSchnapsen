import numpy as np
from schnapsen.bots.rand import RandBot
from src.schnapsen.game import Bot, ExchangeTrick, FollowerPerspective, GameState, LeaderPerspective, Marriage, PlayerPerspective, RegularMove, SchnapsenDeckGenerator, Move, Trick, GamePhase, Trump_Exchange
from typing import List, Optional, cast, Literal
from src.schnapsen.deck import Suit, Rank
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import time
import pathlib
from ast import List
from src.schnapsen.game import Bot, GamePhase, PlayerPerspective, Move, SchnapsenDeckGenerator, SchnapsenTrickScorer, Score
from src.schnapsen.deck import Suit, Card, Rank
from typing import Optional

from .rdeep import RdeepBot

from .ml_bot import get_move_feature_vector, get_one_hot_encoding_of_card_suit, get_state_feature_vector




import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv2D,Conv1D
from keras.optimizers import Adam
import keras.initializers
class DQNAgent(Bot):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.995    # discount rate
        self.epsilon = 0.995  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.batch_size = 32
        self.visited = True
        self.rdeeptimes = 0
        self.nnmoves =0
        self.marriage_count = 0
        self.exchange_count = 0
        self.possible_marriage_count = 0
        self.possible_exchange_count = 0
        self.predicted_marriage_count = 0
        self.predicted_exchange_count = 0
        self.neurons1 = 5
        self.neurons2 = 5

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        initializer = keras.initializers.HeNormal()
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        #model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done,valid_moves,next_valid_moves):
        self.memory.append((state, action, reward, next_state, done,valid_moves,next_valid_moves))

    def act(self, state,leader_move: Optional[Move]):
        normal_state = state
        valid_moves = state.valid_moves()

        self.possible_marriage_count += sum(1 for move in valid_moves if move.is_marriage())
        self.possible_exchange_count += sum(1 for move in valid_moves if move.is_trump_exchange())

        valid_int_moves = [self.move_to_int(move) for move in valid_moves]

        state = get_state_feature_vector(state)
        state = np.reshape(state, [1, 133])


        if np.random.rand() <= self.epsilon:
            #Pick a random move or a move using a different bot strategy

            '''trainer = RdeepBot(num_samples=16, depth=4, rand=random.Random(4564654644))'''
            trainer = RandBot(464566)
            self.rdeeptimes+=1
            return self.move_to_int(trainer.get_move(normal_state,leader_move))
        else:
            #Pick a move using the neural network

            act_values = self.model.predict(state,verbose=0)[0]
            legal_q_values = [act_values[i] for i in valid_int_moves]
            action = [np.argmax(legal_q_values)]
            self.nnmoves+=1

            return valid_int_moves[action[0]]

    def get_valid(self,state,valid_moves):

            #Pick a move using the neural network and also validating it
        
            valid_int_moves = [self.move_to_int(move) for move in valid_moves]
            act_values = self.model.predict(state,verbose=0)[0]
            legal_q_values = [act_values[i] for i in valid_int_moves]
            print(legal_q_values)

            return [legal_q_values]

            
    
        
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done,valid_moves,next_valid_moves in minibatch:
            target = self.model.predict(state,verbose=0)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state,verbose=0)[0]
                t = self.target_model.predict(next_state,verbose=0)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def int_to_move(self,move: int):
        #ALL CARDS:
        ACE_DIAMONDS = Card.get_card(Rank.ACE, Suit.DIAMONDS)
        TEN_DIAMONDS = Card.get_card(Rank.TEN, Suit.DIAMONDS)
        JACK_DIAMONDS = Card.get_card(Rank.JACK, Suit.DIAMONDS)
        QUEEN_DIAMONDS = Card.get_card(Rank.QUEEN, Suit.DIAMONDS)
        KING_DIAMONDS = Card.get_card(Rank.KING, Suit.DIAMONDS)

        ACE_HEARTS = Card.get_card(Rank.ACE, Suit.HEARTS)
        TEN_HEARTS = Card.get_card(Rank.TEN, Suit.HEARTS)
        JACK_HEARTS = Card.get_card(Rank.JACK, Suit.HEARTS)
        QUEEN_HEARTS = Card.get_card(Rank.QUEEN, Suit.HEARTS)
        KING_HEARTS = Card.get_card(Rank.KING, Suit.HEARTS)

        ACE_SPADES = Card.get_card(Rank.ACE, Suit.SPADES)
        TEN_SPADES = Card.get_card(Rank.TEN, Suit.SPADES)
        JACK_SPADES = Card.get_card(Rank.JACK, Suit.SPADES)
        QUEEN_SPADES = Card.get_card(Rank.QUEEN, Suit.SPADES)
        KING_SPADES = Card.get_card(Rank.KING, Suit.SPADES)

        ACE_CLUBS = Card.get_card(Rank.ACE, Suit.CLUBS)
        TEN_CLUBS = Card.get_card(Rank.TEN, Suit.CLUBS)
        JACK_CLUBS = Card.get_card(Rank.JACK, Suit.CLUBS)
        QUEEN_CLUBS = Card.get_card(Rank.QUEEN, Suit.CLUBS)
        KING_CLUBS = Card.get_card(Rank.KING, Suit.CLUBS)

        all_cards = []
        all_cards.append(ACE_DIAMONDS)
        all_cards.append(TEN_DIAMONDS)
        all_cards.append(JACK_DIAMONDS)
        all_cards.append(QUEEN_DIAMONDS)
        all_cards.append(KING_DIAMONDS)

        all_cards.append(ACE_HEARTS)
        all_cards.append(TEN_HEARTS)
        all_cards.append(JACK_HEARTS)
        all_cards.append(QUEEN_HEARTS)
        all_cards.append(KING_HEARTS)

        all_cards.append(ACE_SPADES)
        all_cards.append(TEN_SPADES)
        all_cards.append(JACK_SPADES)
        all_cards.append(QUEEN_SPADES)
        all_cards.append(KING_SPADES)

        all_cards.append(ACE_CLUBS)
        all_cards.append(TEN_CLUBS)
        all_cards.append(JACK_CLUBS)
        all_cards.append(QUEEN_CLUBS)
        all_cards.append(KING_CLUBS)

        cards_in_hand = all_cards

        valid_moves: list[Move] = [RegularMove(card) for card in cards_in_hand]

        #Marriages:

        valid_moves.append(Marriage(QUEEN_DIAMONDS, KING_DIAMONDS))
        valid_moves.append(Marriage(QUEEN_HEARTS, KING_HEARTS))
        valid_moves.append(Marriage(QUEEN_SPADES, KING_SPADES))
        valid_moves.append(Marriage(QUEEN_CLUBS, KING_CLUBS))

        #Trump exchanges:

        valid_moves.append(Trump_Exchange(JACK_DIAMONDS))
        valid_moves.append(Trump_Exchange(JACK_HEARTS))
        valid_moves.append(Trump_Exchange(JACK_SPADES))
        valid_moves.append(Trump_Exchange(JACK_CLUBS))

        return valid_moves[move]
    def move_to_int(self,move: Move):

        #ALL CARDS:
        ACE_DIAMONDS = Card.get_card(Rank.ACE, Suit.DIAMONDS)
        TEN_DIAMONDS = Card.get_card(Rank.TEN, Suit.DIAMONDS)
        JACK_DIAMONDS = Card.get_card(Rank.JACK, Suit.DIAMONDS)
        QUEEN_DIAMONDS = Card.get_card(Rank.QUEEN, Suit.DIAMONDS)
        KING_DIAMONDS = Card.get_card(Rank.KING, Suit.DIAMONDS)

        ACE_HEARTS = Card.get_card(Rank.ACE, Suit.HEARTS)
        TEN_HEARTS = Card.get_card(Rank.TEN, Suit.HEARTS)
        JACK_HEARTS = Card.get_card(Rank.JACK, Suit.HEARTS)
        QUEEN_HEARTS = Card.get_card(Rank.QUEEN, Suit.HEARTS)
        KING_HEARTS = Card.get_card(Rank.KING, Suit.HEARTS)

        ACE_SPADES = Card.get_card(Rank.ACE, Suit.SPADES)
        TEN_SPADES = Card.get_card(Rank.TEN, Suit.SPADES)
        JACK_SPADES = Card.get_card(Rank.JACK, Suit.SPADES)
        QUEEN_SPADES = Card.get_card(Rank.QUEEN, Suit.SPADES)
        KING_SPADES = Card.get_card(Rank.KING, Suit.SPADES)

        ACE_CLUBS = Card.get_card(Rank.ACE, Suit.CLUBS)
        TEN_CLUBS = Card.get_card(Rank.TEN, Suit.CLUBS)
        JACK_CLUBS = Card.get_card(Rank.JACK, Suit.CLUBS)
        QUEEN_CLUBS = Card.get_card(Rank.QUEEN, Suit.CLUBS)
        KING_CLUBS = Card.get_card(Rank.KING, Suit.CLUBS)

        all_cards = []
        all_cards.append(ACE_DIAMONDS)
        all_cards.append(TEN_DIAMONDS)
        all_cards.append(JACK_DIAMONDS)
        all_cards.append(QUEEN_DIAMONDS)
        all_cards.append(KING_DIAMONDS)

        all_cards.append(ACE_HEARTS)
        all_cards.append(TEN_HEARTS)
        all_cards.append(JACK_HEARTS)
        all_cards.append(QUEEN_HEARTS)
        all_cards.append(KING_HEARTS)

        all_cards.append(ACE_SPADES)
        all_cards.append(TEN_SPADES)
        all_cards.append(JACK_SPADES)
        all_cards.append(QUEEN_SPADES)
        all_cards.append(KING_SPADES)

        all_cards.append(ACE_CLUBS)
        all_cards.append(TEN_CLUBS)
        all_cards.append(JACK_CLUBS)
        all_cards.append(QUEEN_CLUBS)
        all_cards.append(KING_CLUBS)

        cards_in_hand = all_cards

        valid_moves: list[Move] = [RegularMove(card) for card in cards_in_hand]

        #Marriages:

        valid_moves.append(Marriage(QUEEN_DIAMONDS, KING_DIAMONDS))
        valid_moves.append(Marriage(QUEEN_HEARTS, KING_HEARTS))
        valid_moves.append(Marriage(QUEEN_SPADES, KING_SPADES))
        valid_moves.append(Marriage(QUEEN_CLUBS, KING_CLUBS))

        #Trump exchanges:

        valid_moves.append(Trump_Exchange(JACK_DIAMONDS))
        valid_moves.append(Trump_Exchange(JACK_HEARTS))
        valid_moves.append(Trump_Exchange(JACK_SPADES))
        valid_moves.append(Trump_Exchange(JACK_CLUBS))

        for move1 in valid_moves:
            if move1 == move:
                return valid_moves.index(move)

        return -1 

    def reward(self,state: PlayerPerspective, action : Move, Nextstate: PlayerPerspective,won: bool):
        reward = 0

        #Reward for special moves:

        if type(action) == Marriage:
            reward+= 20
        elif type(action) == Trump_Exchange:
            reward+= 10

        ranker = SchnapsenTrickScorer()

        #Reward for winning a trick:

        if len(Nextstate.get_won_cards()) > len(state.get_won_cards()):
            new_cards = set(Nextstate.get_won_cards()).difference(state.get_won_cards())
            for card in new_cards:
                reward += ranker.rank_to_points(card.rank)
            return reward
        else :
            new_cards = set(Nextstate.get_opponent_won_cards()).difference(state.get_opponent_won_cards())
            for card in new_cards:
                reward -= ranker.rank_to_points(card.rank)
            

        #Optional: Reward for winning the game:

        '''if won:
            print("Reward for winning the game: ", 1)
            reward += 66
        else:
            #print("Reward for loosing the game: ", -1)
            #reward -= 66
        print(reward)'''

        return reward
        

    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        

        action = self.act(state,leader_move=leader_move)
        action = self.int_to_move(action)
        
        if (len(self.memory) > self.batch_size) and (self.visited == True):
            self.replay(self.batch_size)
            self.visited = False
            
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        return action
    

   

        
        
      
  


    
        
    



    
