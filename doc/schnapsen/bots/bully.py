from example_bot import ExampleBot

class Bully(ExampleBot):
    def __init__(self, name):
        super().__init__(name)

    def play(self, game_state, round_state):
        return self._choose_card(game_state, round_state)

    def _choose_card(self, game_state, round_state):
        if round_state['trick'] == []:
            return self._choose_first_card(game_state, round_state)
        else:
            return self._choose_next_card(game_state, round_state)

