import os.path
import random
from typing import Optional


import click
import gym
from schnapsen.bots import MLDataBot, train_ML_model, MLPlayingBot, BullyBot, RandBot, AlphaBetaBot, RdeepBot,SecondBot,QAgent

from DeepL import agent
import numpy as np


from schnapsen.bots.example_bot import ExampleBot

from schnapsen.game import (Bot, Move, PlayerPerspective,
                            SchnapsenGamePlayEngine, Trump_Exchange)
from schnapsen.twenty_four_card_schnapsen import \
    TwentyFourSchnapsenGamePlayEngine

from schnapsen.bots.rdeep import RdeepBot

engine = SchnapsenGamePlayEngine()


def create_state(self, state: PlayerPerspective) -> np.ndarray:
    state.get_hand()


def play_test_game():
    bot1 = BullyBot()
    bot2 = QAgent()
    winner, points, score = engine.play_game(bot1, bot2, random.Random(1))
    print(winner)


play_test_game()


