from src.schnapsen.game import Bot, PlayerPerspective, Move, SchnapsenTrickScorer, Score
from src.schnapsen.deck import Suit, Card, Rank
from typing import Optional


class SecondBot(Bot):
    """
    This Bot is here to serve as an example of the different methods the PlayerPerspective provides.
    In the end it is just playing the first valid move.
    """
    prevMove = None

    def __init__(self) -> None:
        super().__init__()
        self.prevMove = None

    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        # You can get information on the state from your perspective
   
        # Get valid moves
        moves: list[Move] = state.valid_moves()
        one_move: Move = moves[0]
        possibleMoves =[]
        #print(state.get_opponent_score().direct_points)
        if state.get_opponent_score().direct_points > state.get_my_score().direct_points:
            for move in moves:
                if move.is_marriage() or move.is_trump_exchange():
                    one_move = move
                    self.prevMove = move
                    break
        else:
            
            for move in moves:
              for card in move.__getattribute__("cards"):
                if self.prevMove is not None:
                    if card.suit == self.prevMove.__getattribute__("cards")[0].suit:
                        possibleMoves.append(move)
                        #one_move = move
            Low = 11
            for moves in possibleMoves:
                if SchnapsenTrickScorer.rank_to_points(self,moves.__getattribute__("cards")[0].rank) <= Low:
                    max = SchnapsenTrickScorer.rank_to_points(self,moves.__getattribute__("cards")[0].rank)
                    one_move = moves
        if possibleMoves is None:
            one_move = moves[0]
        return one_move