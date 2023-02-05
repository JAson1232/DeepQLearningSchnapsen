"""Create a bot in a separate .py and import them here, so that one can simply import
it by from schnapsen.bots import MyBot.
"""
from .Bully import BullyBot
from .SecondBot import SecondBot
from .rand import RandBot
from .alphabeta import AlphaBetaBot
from .rdeep import RdeepBot
from .ml_bot import MLDataBot, MLPlayingBot, train_ML_model
from.DeepQAgent import DQNAgent
from .gui.guibot import SchnapsenServer
from .late import LateBot


__all__ = ["RandBot", "AlphaBetaBot", "RdeepBot", "MLDataBot", "MLPlayingBot", "train_ML_model", "SchnapsenServer","BullyBot","SecondBot","DQNAgent","LateBot"]
