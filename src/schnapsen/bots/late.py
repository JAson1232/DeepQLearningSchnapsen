from ast import List
from src.schnapsen.game import Bot, GamePhase, PlayerPerspective, Move, SchnapsenDeckGenerator, SchnapsenTrickScorer, Score
from src.schnapsen.deck import Suit, Card, Rank
from typing import Optional




class LateBot(Bot):
    """
    This Bot is here to serve as an example of the different methods the PlayerPerspective provides.
    In the end it is just playing the first valid move.
    """
    

    def get_move(self, state: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        
        # Get valid moves
        moves: list[Move] = state.valid_moves()
        one_move: Move = moves[0]
    
        for move in moves:
            max = 0
        
            for cards in move.__getattribute__("cards"):
                #print(cards.suit)
                if cards.suit == state.get_trump_suit():
                    one_move = move
                    break
                elif leader_move is not None:
                    #print("I am not the leader")
                    #print(leader_move.cards[0])
                    if cards.suit == leader_move.cards[0].suit:
                        one_move = move
                        break
                else:
                    #print(SchnapsenTrickScorer.rank_to_points(self,cards.rank))
                    if SchnapsenTrickScorer.rank_to_points(self,cards.rank) > max:
                        max = SchnapsenTrickScorer.rank_to_points(self,cards.rank)
                        one_move = move
                break
        print("Bully Move: ")
        print(one_move)
        return one_move