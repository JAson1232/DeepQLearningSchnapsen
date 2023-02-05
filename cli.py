from keras.models import load_model
import os.path
import random
from typing import Optional

import os
import click

from src.schnapsen.bots import MLDataBot, train_ML_model, MLPlayingBot, BullyBot, RandBot, AlphaBetaBot, RdeepBot,SecondBot,DQNAgent,LateBot
from tqdm import tqdm
import numpy as np

from src.schnapsen.bots.ml_bot import get_move_feature_vector, get_one_hot_encoding_of_card_suit, get_state_feature_vector
from src.schnapsen.bots.example_bot import ExampleBot

from src.schnapsen.game import (Bot, Move, PlayerPerspective,
                            SchnapsenGamePlayEngine, Trump_Exchange)
from src.schnapsen.twenty_four_card_schnapsen import \
    TwentyFourSchnapsenGamePlayEngine

from src.schnapsen.bots.rdeep import RdeepBot



@click.group()
def main() -> None:
    """Various Schnapsen Game Examples"""


class RandBot(Bot):
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(self.seed)

    def get_move(self, player_perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        moves = player_perspective.valid_moves()
        move = self.rng.choice(list(moves))
        return move

    def __repr__(self) -> str:
        return f"RandBot(seed={self.seed})"


@main.command()
def random_game() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = RandBot(12112121)
    bot2 = RandBot(464566)
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


class NotificationExampleBot(Bot):

    def get_move(self, player_perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        moves = player_perspective.valid_moves()
        return moves[0]

    def notify_game_end(self, won: bool, state: PlayerPerspective) -> None:
        print(f'result {"win" if won else "lost"}')
        print(f'I still have {len(state.get_hand())} cards left')

    def notify_trump_exchange(self, move: Trump_Exchange) -> None:
        print(f"That trump exchanged! {move.jack}")


@main.command()
def notification_game() -> None:
    engine = TwentyFourSchnapsenGamePlayEngine()
    bot1 = NotificationExampleBot()
    bot2 = RandBot(464566)
    engine.play_game(bot1, bot2, random.Random(94))


class HistoryBot(Bot):
    def get_move(self, player_perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        history = player_perspective.get_game_history()
        print(f'the initial state of this game was {history[0][0]}')
        moves = player_perspective.valid_moves()
        return moves[0]


@main.group()
def ml() -> None:
    """Commands for the ML bot"""


@ml.command()
def create_replay_memory_dataset() -> None:
    # define replay memory database creation parameters
    num_of_games: int = 1000
    replay_memory_dir: str = 'ML_replay_memories'
    replay_memory_filename: str = 'test_replay_memory.txt'
    bot_1_behaviour = RandBot(5234243)
    bot_2_behaviour = RandBot(54354)
    random_seed: int = 1
    delete_existing_older_dataset = True

    # check if needed to delete any older versions of the dataset
    replay_memory_file_path = os.path.join(replay_memory_dir, replay_memory_filename)
    if delete_existing_older_dataset and os.path.exists(replay_memory_file_path):
        print(f"An existing dataset was found at location '{replay_memory_file_path}', which will be deleted as selected.")
        os.remove(replay_memory_file_path)

    # in any case make sure the directory exists
    if not os.path.exists(replay_memory_dir):
        os.mkdir(replay_memory_dir)

    # create new replay memory dataset, according to the behaviour of the provided bots and the provided random seed
    engine = SchnapsenGamePlayEngine()
    replay_memory_recording_bot_1 = MLDataBot(bot_1_behaviour, replay_memory_file_path=replay_memory_file_path)
    replay_memory_recording_bot_2 = MLDataBot(bot_2_behaviour, replay_memory_file_path=replay_memory_file_path)
    for i in range(num_of_games):
        engine.play_game(replay_memory_recording_bot_1, replay_memory_recording_bot_2, random.Random(random_seed))
    print(f"Replay memory dataset recorder for {num_of_games} games.\nDataset is stored at: {replay_memory_file_path}")


@ml.command()
def train_model() -> None:
    replay_memory_filename = 'test_replay_memory.txt'
    replay_memories_directory = 'ML_replay_memories'
    model_name = 'test_model'
    model_dir = "ML_models"
    overwrite = False

    train_ML_model(replay_memory_filename=replay_memory_filename, replay_memories_directory=replay_memories_directory,
                   model_name=model_name, model_dir=model_dir, overwrite=overwrite)


@ml.command()
def try_bot_game() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = MLPlayingBot(model_name='test_model', model_dir="ML_models")
    bot2 = RandBot(464566)
    winner, points, score = engine.play_game(bot1, bot2, random.Random(1))
    print(f"Winner is: {winner}, with {points} points!")


@main.command()
def game_24() -> None:
    engine = TwentyFourSchnapsenGamePlayEngine()
    bot1 = RandBot(12112121)
    bot2 = RandBot(464566)
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


@main.command()
def rdeep_game() -> None:
    bot1: Bot
    bot2: Bot
    engine = SchnapsenGamePlayEngine()
    rdeep = bot1 = RdeepBot(num_samples=16, depth=4, rand=random.Random(4564654644))
    bot2 = RandBot(464566)
    wins = 0
    amount = 101
    for i in range(amount):
        if i % 2 == 0:
            bot1, bot2 = bot2, bot1
        winner_id, _, _ = engine.play_game(bot1, bot2, random.Random(5))
        if winner_id == rdeep:
            wins += 1
        if i > 0 and i % 10 == 0:
            print(f"won {wins} out of {i}")


@main.command()
def try_example_bot_game() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = ExampleBot()
    bot2 = RandBot(464566)
    winner, points, score = engine.play_game(bot1, bot2, random.Random(1))
    print(f"Winner is: {winner}, with {points} points!")


@main.command()
def bully_bot() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = BullyBot()
    bot2 = RandBot(464566)
    winner, points, score = engine.play_game(bot1, bot2, random.Random(1))
    print(f"Winner is: {winner}, with {points} points!")
    
@main.command()
def late_bot() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = LateBot()
    bot2 = RandBot(464566)
    winner, points, score = engine.play_game(bot1, bot2, random.Random(1))
    print(f"Winner is: {winner}, with {points} points!")

@main.command()
def train_DQN() -> None:
    bot1 = DQNAgent(133,28)
    bot2 = BullyBot()
    bot3 = RdeepBot(num_samples=16, depth=4, rand=random.Random(4564654644))
    bot4 = RandBot(464566)
    bots = [bot2, bot3, bot4]
    training_cycles = [[DQNAgent(133,28),bot2,bot3,bot4],[DQNAgent(133,28),bot2], [DQNAgent(133,28),bot3], [DQNAgent(133,28),bot4],[DQNAgent(133,28),bot2,bot3],[DQNAgent(133,28),bot2,bot4],[DQNAgent(133,28),bot3,bot4]]
    names = ["DQN_Bully_Rdeep_Rand","DQN_Bully", "DQN_Rdeep", "DQN_Rand", "DQN_Bully_Rdeep", "DQN_Bully_Rand", "DQN_Rdeep_Rand"]
    training_cycle_index = 0
    for training_cycle in training_cycles:
        games = {}
        rdeeptimes=0
        count = 0
        for i in tqdm(range(500,1000)):
            engine = SchnapsenGamePlayEngine() 
            if len(training_cycle) == 2:
                winner, points, score,history = engine.play_game(training_cycle[1], training_cycle[0], random.Random(i))
            elif len(training_cycle) == 3:
                if i%2 == 0:
                    winner, points, score,history = engine.play_game(training_cycle[1], training_cycle[0], random.Random(i))
                else:
                    winner, points, score,history = engine.play_game(training_cycle[2], training_cycle[0], random.Random(i))
            else:
                if i%3 == 0:
                    winner, points, score,history = engine.play_game(training_cycle[1], training_cycle[0], random.Random(i))
                elif i%3 == 1:
                    winner, points, score,history = engine.play_game(training_cycle[2], training_cycle[0], random.Random(i))
                else:
                    winner, points, score,history = engine.play_game(training_cycle[3], training_cycle[0], random.Random(i))
            #if bot1.epsilon > bot1.epsilon_min:
            #       bot1.epsilon *= bot1.epsilon_decay
            #print(f"Winner is: {winner}, with {points} points!")
            #print(history[0][0]) # Current State
            #print("\n")
            #print(history[0][1]) # Current Action
            #print("\n")
            #print(history[1][0]) # Next State
            
            for i in range(0,len(history)-1):
                if history[i][0].am_i_leader():
                        
                        if history[i][1].is_trump_exchange():
                            #print("Exchange trick")
                            action1 = history[i][1]._cards()
                        else:
                            action1 = history[i][1].leader_move
                else:
                        if history[i][1].is_trump_exchange():
                            #print("Exchange trick")
                            action1 = history[i][1]._cards()
                        else:
                            action1 = history[i][1].follower_move
                #for i in range(0,len(history)):
                    #training_cycle[0].reward(history[i],)
                if type(winner) == DQNAgent:
                    reward = (training_cycle[0].reward(history[i][0],action1,history[i+1][0],True))
                else:
                    reward = (training_cycle[0].reward(history[i][0],action1,history[i+1][0],False))
                if history[i][1] == None:
                    training_cycle[0].remember(np.reshape(get_state_feature_vector(history[i][0]),[1, 133]), training_cycle[0].move_to_int(action1), reward, np.reshape(get_state_feature_vector(history[i+1][0]), [1, 133]),True,history[i][0].valid_moves(),history[i+1][0].valid_moves())
                else:
                    training_cycle[0].remember(np.reshape(get_state_feature_vector(history[i][0]),[1, 133]), training_cycle[0].move_to_int(action1), reward, np.reshape(get_state_feature_vector(history[i+1][0]), [1, 133]),False,history[i][0].valid_moves(),history[i+1][0].valid_moves())
            
            if count % 25 == 0:
                training_cycle[0].update_target_model()
            
            if count % 1==0:
                training_cycle[0].visited = True
                
            #games[str(winner)] = games.get(str(winner), 0) + 1
            count += 1
       

        #Save Model
        os.mkdir(names[training_cycle_index])
        training_cycle[0].target_model.save(os.path.join(str(names[training_cycle_index]),str(names[training_cycle_index] +".h5")))
        
        games = {}
        bot1 = DQNAgent(133,28)
        bot1.model = load_model(os.path.join(str(names[training_cycle_index]),str(names[training_cycle_index] +".h5")))
        bot1.epsilon = 0

        for i in tqdm(range( 0,101)):
            engine = SchnapsenGamePlayEngine()
            winner, points, score,history = engine.play_game(BullyBot(), bot1, random.Random(i))
            games[type(winner)] = games.get(type(winner), 0) + 1
            
        #Save Results
        lines = []
        lines.append(str("Games: " + str(games)))
        lines.append(str("Training Moves: " + str(bot1.rdeeptimes)))
        lines.append(str("NN Moves: " + str(bot1.nnmoves)))
        lines.append(str("Number of Marriages: " + str(bot1.marriage_count)))
        lines.append(str("Number of Exchanges: " + str(bot1.exchange_count)))
        lines.append(str("Possible Marriages: " + str(bot1.possible_marriage_count)))
        lines.append(str("Possible Exchanges: "+ str(bot1.possible_exchange_count)))
        lines.append(str("Predicted Marriages: " + str(bot1.predicted_marriage_count)))
        lines.append(str("Predicted Exchanges: " + str(bot1.predicted_exchange_count)))
        lines.append(str("Learning Rate: " + str(bot1.learning_rate)))
        lines.append(str("Epsilon: " + str(bot1.epsilon)))
        lines.append(str("Epsilon Decay: " + str(bot1.epsilon_decay)))
        lines.append(str("Epsilon Min: " + str(bot1.epsilon_min)))
        lines.append(str("Gamma: " + str(bot1.gamma)))
        lines.append(str("Batch Size: " + str(bot1.batch_size)))
        lines.append(str("Neuron 1st Layer: " + str(bot1.neurons1)))
        lines.append(str("Neuron 2nd Layer: " + str(bot1.neurons2)))
        with open (os.path.join(str(names[training_cycle_index]),str(names[training_cycle_index] +"vs.Bully.txt")), 'w') as file:  
            for line_1 in lines:  
                file.write(line_1)  
                file.write('\n')  

        
        bot1 = DQNAgent(133,28)
        bot1.model = load_model(os.path.join(str(names[training_cycle_index]),str(names[training_cycle_index] +".h5")))
        bot1.epsilon = 0
        games = {}

        for i in tqdm(range( 0,101)):
            engine = SchnapsenGamePlayEngine()
            winner, points, score ,history= engine.play_game(RdeepBot(num_samples=16, depth=4, rand=random.Random(4564654644)), bot1, random.Random(i))
            games[type(winner)] = games.get(type(winner), 0) + 1
            
        #Save Results
        lines = []
        lines.append(str("Games: " + str(games)))
        lines.append(str("Training Moves: " + str(bot1.rdeeptimes)))
        lines.append(str("NN Moves: " + str(bot1.nnmoves)))
        lines.append(str("Number of Marriages: " + str(bot1.marriage_count)))
        lines.append(str("Number of Exchanges: " + str(bot1.exchange_count)))
        lines.append(str("Possible Marriages: " + str(bot1.possible_marriage_count)))
        lines.append(str("Possible Exchanges: "+ str(bot1.possible_exchange_count)))
        lines.append(str("Predicted Marriages: " + str(bot1.predicted_marriage_count)))
        lines.append(str("Predicted Exchanges: " + str(bot1.predicted_exchange_count)))
        lines.append(str("Learning Rate: " + str(bot1.learning_rate)))
        lines.append(str("Epsilon: " + str(bot1.epsilon)))
        lines.append(str("Epsilon Decay: " + str(bot1.epsilon_decay)))
        lines.append(str("Epsilon Min: " + str(bot1.epsilon_min)))
        lines.append(str("Gamma: " + str(bot1.gamma)))
        lines.append(str("Batch Size: " + str(bot1.batch_size)))
        lines.append(str("Neuron 1st Layer: " + str(bot1.neurons1)))
        lines.append(str("Neuron 2nd Layer: " + str(bot1.neurons2)))
        with open (os.path.join(str(names[training_cycle_index]),str(names[training_cycle_index] +"vs.RDeep.txt")), 'w') as file:  
            for line_1 in lines:  
                file.write(line_1)  
                file.write('\n')  
        
        games = {}
        bot1 = DQNAgent(133,28)
        bot1.model = load_model(os.path.join(str(names[training_cycle_index]),str(names[training_cycle_index] +".h5")))
        bot1.epsilon = 0

        for i in tqdm(range( 0,101)):
            engine = SchnapsenGamePlayEngine()
            winner, points, score ,history= engine.play_game(RandBot(464566), bot1, random.Random(i))
            games[type(winner)] = games.get(type(winner), 0) + 1
            
        #Save Results
        lines = []
        lines.append(str("Games: " + str(games)))
        lines.append(str("Training Moves: " + str(bot1.rdeeptimes)))
        lines.append(str("NN Moves: " + str(bot1.nnmoves)))
        lines.append(str("Number of Marriages: " + str(bot1.marriage_count)))
        lines.append(str("Number of Exchanges: " + str(bot1.exchange_count)))
        lines.append(str("Possible Marriages: " + str(bot1.possible_marriage_count)))
        lines.append(str("Possible Exchanges: "+ str(bot1.possible_exchange_count)))
        lines.append(str("Predicted Marriages: " + str(bot1.predicted_marriage_count)))
        lines.append(str("Predicted Exchanges: " + str(bot1.predicted_exchange_count)))
        lines.append(str("Learning Rate: " + str(bot1.learning_rate)))
        lines.append(str("Epsilon: " + str(bot1.epsilon)))
        lines.append(str("Epsilon Decay: " + str(bot1.epsilon_decay)))
        lines.append(str("Epsilon Min: " + str(bot1.epsilon_min)))
        lines.append(str("Gamma: " + str(bot1.gamma)))
        lines.append(str("Batch Size: " + str(bot1.batch_size)))
        lines.append(str("Neuron 1st Layer: " + str(bot1.neurons1)))
        lines.append(str("Neuron 2nd Layer: " + str(bot1.neurons2)))
        with open (os.path.join(str(names[training_cycle_index]),str(names[training_cycle_index] +"vs.Random.txt")), 'w') as file:  
            for line_1 in lines:  
                file.write(line_1)  
                file.write('\n')  
        

        training_cycle_index += 1


@main.command()
def play_DQN() -> None:
    games = {}
    bot2 = DQNAgent(133,28)
    bot1 = BullyBot()
    #bot1 = RdeepBot(num_samples=16, depth=4, rand=random.Random(4564654644))
    bot2.target_model = load_model("DecentModel.h5")
    
    for i in range( 0,101):
        engine = SchnapsenGamePlayEngine()
        
        
        winner, points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Winner is: {winner}, with {points} points!")
        games[type(winner)] = games.get(type(winner), 0) + 1
        if type(winner) == DQNAgent:
            print(bot2.epsilon)

    print(games)

    print("DQN Agent Results:")
    print("Training Moves: ",bot2.rdeeptimes)
    print("NN Moves: ",bot2.nnmoves)
    print("Number of Marriages: ",bot2.marriage_count)
    print("Number of Exchanges: ",bot2.exchange_count)
    print("Possible Marriages: ",bot2.possible_marriage_count)
    print("Possible Exchanges: ",bot2.possible_exchange_count)



if __name__ == "__main__":
    main()
