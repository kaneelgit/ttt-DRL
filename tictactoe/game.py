class TicTacToe:
    def __init__(self):
        self.rows = 3
        self.columns = 3
        self.board = [[' ' for _ in range(self.columns)] for _ in range(self.rows)]
        self.current_player = 'X'
        self.winner = None

    def print_board(self):
        for row in self.board:
            print("| " + " | ".join(row) + " |")

    def is_move_valid(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.columns and self.board[row][col] == ' '

    def make_move(self, row, col):
        if not self.is_move_valid(row, col):
            return False

        self.board[row][col] = self.current_player
        return True

    def check_winner(self):
        for i in range(self.rows):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                self.winner = self.board[i][0]
                return True

            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                self.winner = self.board[0][i]
                return True

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            self.winner = self.board[0][0]
            return True

        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            self.winner = self.board[0][2]
            return True

        return False

    def play_game(self):
        while not self.winner:
            self.print_board()
            row = int(input(f"Player {self.current_player}, enter row (1-{self.rows}): ")) - 1
            col = int(input(f"Player {self.current_player}, enter column (1-{self.columns}): ")) - 1
            if self.make_move(row, col):
                if self.check_winner():
                    self.print_board()
                    print(f"Player {self.current_player} wins!")
                    break
                if all(cell != ' ' for row in self.board for cell in row):
                    self.print_board()
                    print("It's a tie!")
                    break
                self.current_player = 'O' if self.current_player == 'X' else 'X'

    def play_move(self, row, col, print = True):
        '''
        Play move manually.
        '''
        if self.make_move(row, col):
            if self.check_winner():
                if print:
                    self.print_board()
                return self.winner

            if all(cell != ' ' for row in self.board for cell in row):
                if print:
                    self.print_board()
                return 'draw'

            self.current_player = 'O' if self.current_player == 'X' else 'X'
            if print:
                self.print_board()


if __name__ == "__main__":
    c4 = TicTacToe()
    c4.play_game()