import random
from schnapsen.bots import SchnapsenServer
from schnapsen.bots import RandBot, AlphaBetaBot, RdeepBot,SecondBot,BullyBot,DQNAgent

from schnapsen.game import SchnapsenGamePlayEngine, Bot
import click
from keras.models import load_model



@click.command()
@click.option('--bot', '-b',
              type=click.Choice(['AlphaBetaBot', 'RdeepBot', 'MLDataBot', 'MLPlayingBot', 'RandBot',"SecondBot","BullyBot","DQNAgent"], case_sensitive=False),
              default='DQNAgent', help="The bot you want to play against.")
def main(bot: str) -> None:
    """Run the GUI."""
    engine = SchnapsenGamePlayEngine()
    bot1: Bot
    print(bot.lower())
    with SchnapsenServer() as s:
        if bot.lower() == "randbot":
            bot1 = RandBot(12)
            print("RandBot")
        elif bot.lower() in ["alphabeta", "alphabetabot"]:
            bot1 = AlphaBetaBot()
            print("AlphaBetaBot")
        elif bot.lower() == "rdeepbot":
            bot1 = RdeepBot(num_samples=16, depth=4, rand=random.Random(42))
            print("RdeepBot")
        elif bot.lower() == "bullybot":
            bot1 = BullyBot()
            print("BullyBot")
        elif bot.lower() == "dqnagent":
            bot1 = DQNAgent(133,28)
            bot1.epsilon = 0
            bot1.model = load_model("schnapsen_model.h5")
            print("BullyBot")
        else:
            raise NotImplementedError
        bot2 = s.make_gui_bot(name="BullyBot")
        # bot1 = s.make_gui_bot(name="mybot1")
        engine.play_game(bot1, bot2, random.Random(35))


if __name__ == "__main__":
    main()
