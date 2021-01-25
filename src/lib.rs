use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Player {
    Max,
    Min,
}

impl Player {
    /// The worst possible score for the player.
    /// For max this is -infinity, For Min this is infinity.
    pub fn worst(&self) -> i16 {
        match self {
            Self::Max => i16::MIN,
            Self::Min => i16::MAX,
        }
    }

    /// Return a function where f(x,y) tells you if
    /// x is a better score then y. relative to the player.
    pub fn better(&self) -> fn(i16, i16) -> bool {
        match self {
            Self::Max => |x, y| x > y,
            Self::Min => |x, y| x < y,
        }
    }

    /// Return the best of x, y (relative to the current player)
    pub fn best(&self, x: i16, y: i16) -> i16 {
        match self {
            Self::Max => i16::max(x, y),
            Self::Min => i16::min(x, y),
        }
    }

    /// Switch sides, Max -> Min and Min -> Max
    pub fn other(self) -> Self {
        match self {
            Self::Max => Self::Min,
            Self::Min => Self::Max,
        }
    }
}

/// An abstract, 2 player board game
pub trait BoardGame {
    type Move;

    /// Return the player who's turn it is,
    /// Max is trying to maximize `self.result`/`self.score` and
    /// Min is trying to minimize. In chess this would be white and black.
    fn turn(&self) -> Player;

    /// Return the starting position
    fn start() -> Self;

    /// Return all legal moves for the current board state.
    fn legal(&self) -> Vec<Self::Move>;

    /// Make a legal move on the board
    fn make_move_mut(&mut self, mv: Self::Move);

    /// Undo the last move, an undo when there is no history
    /// should panic.
    fn undo_move_mut(&mut self);

    /// Return an evaluation of the result if the game is over.
    fn result(&self) -> Option<i16>;

    /// Return an optional score, if known,
    /// This can be an estimate in games like chess.
    /// By default this uses `self.result`.
    fn score(&self) -> Option<i16> {
        self.result()
    }
}

// TODO: use generics to implement helpers around BoardGame (eg, make_move)

/// The minimax algorithm, doing a full search and using `BoardGame.result` for score.
pub fn minimax<T>(board: &mut T) -> i16
where
    T: BoardGame,
{
    if let Some(score) = board.result() {
        return score;
    }

    let player = board.turn();
    let mut score = player.worst();

    let legal_moves = board.legal();
    assert!(legal_moves.len() > 0, "result is None, but legal is []");
    for mv in legal_moves {
        board.make_move_mut(mv);
        let mv_score = minimax(board);
        score = player.best(score, mv_score);

        board.undo_move_mut();
    }

    return score;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[rustfmt::skip]
pub enum TicTacToeSquare {
    A3,B3,C3,
    A2,B2,C2,
    A1,B1,C1,
}

impl TicTacToeSquare {
    fn from_usize(n: usize) -> TicTacToeSquare {
        use TicTacToeSquare::*;

        match n {
            0 => A3,
            1 => B3,
            2 => C3,
            3 => A2,
            4 => B2,
            5 => C2,
            6 => A1,
            7 => B1,
            8 => C1,
            _ => panic!("cannot convert {} to TicTacToeSquare", n),
        }
    }
}

type TicTacToeMove = TicTacToeSquare;

#[rustfmt::skip]
#[derive(Debug, PartialEq, Eq)]
enum TicTacToeResult { WinX, WinO, Draw}

impl TicTacToeResult {
    fn score(&self) -> i16 {
        use TicTacToeResult::*;

        match self {
            WinX => 1,
            WinO => -1,
            Draw => 0,
        }
    }
}

pub struct TicTacToe {
    turn: Player,              // The player to move, X=MAX, O=MIN
    moves: Vec<TicTacToeMove>, // Moves made so far, used in undo()
    board: [Option<Player>; 9],
}

impl fmt::Display for TicTacToe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..9 {
            let sq = self.board[i];
            match sq {
                Some(Player::Max) => write!(f, "X")?,
                Some(Player::Min) => write!(f, "O")?,
                None => write!(f, "-")?,
            }

            if (i + 1) % 3 == 0 {
                write!(f, "\n")?;
            }
        }

        writeln!(
            f,
            "side to play: {}",
            if self.turn == Player::Max { "X" } else { "O" }
        )
    }
}

impl BoardGame for TicTacToe {
    type Move = TicTacToeMove;

    fn start() -> Self {
        Self {
            turn: Player::Max,
            moves: Vec::new(),
            board: [None; 9],
        }
    }

    fn turn(&self) -> Player {
        self.turn
    }

    fn make_move_mut(&mut self, mv: Self::Move) {
        self.board[mv as usize] = Some(self.turn);
        self.turn = self.turn.other();
        self.moves.push(mv);
    }

    fn undo_move_mut(&mut self) {
        let mv = self.moves.pop().unwrap();
        self.board[mv as usize] = None;
        self.turn = self.turn.other();
    }

    fn legal(&self) -> Vec<Self::Move> {
        self.board
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_none())
            .map(|(i, _)| TicTacToeSquare::from_usize(i))
            .collect()
    }

    fn result(&self) -> Option<i16> {
        use TicTacToeResult::*;

        const X: Option<Player> = Some(Player::Max);
        const O: Option<Player> = Some(Player::Min);
        let b = self.board;

        // Horizontal check
        for col in 0..3 {
            let s = 3 * col;
            match b[s..s + 3] {
                [X, X, X] => return Some(WinX.score()),
                [O, O, O] => return Some(WinO.score()),
                _ => {}
            };
        }

        // Virtical check
        for row in 0..3 {
            match [b[row], b[row + 3], b[row + 6]] {
                [X, X, X] => return Some(WinX.score()),
                [O, O, O] => return Some(WinO.score()),
                _ => {}
            }
        }

        // Diagonal checks
        match [b[0], b[4], b[8]] {
            [X, X, X] => return Some(WinX.score()),
            [O, O, O] => return Some(WinO.score()),
            _ => {}
        }

        match [b[2], b[4], b[6]] {
            [X, X, X] => return Some(WinX.score()),
            [O, O, O] => return Some(WinO.score()),
            _ => {}
        }

        // Draw check, if filled and no winner then draw
        if b.iter().filter(|x| x.is_some()).count() == 9 {
            return Some(Draw.score());
        }

        // Ongoing
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod tictactoe {
        use super::*;
        use TicTacToeResult::*;
        use TicTacToeSquare::*;
        const X: Option<Player> = Some(Player::Max);
        const O: Option<Player> = Some(Player::Min);

        #[test]
        fn display() {
            let tic = TicTacToe::start();
            eprintln!("{}", tic);
            // panic!("show me the formatting!");
        }

        #[test]
        fn legal_moves() {
            let mut tic = TicTacToe::start();
            assert_eq!(tic.legal(), vec![A3, B3, C3, A2, B2, C2, A1, B1, C1]);

            tic.make_move_mut(A3);
            eprintln!("after A3\n{}", tic);
            assert_eq!(tic.legal(), vec![B3, C3, A2, B2, C2, A1, B1, C1]);

            tic.undo_move_mut();
            assert_eq!(tic.legal(), vec![A3, B3, C3, A2, B2, C2, A1, B1, C1]);
        }

        #[test]
        fn game_result_none() {
            assert_eq!(TicTacToe::start().result(), None);
        }

        #[test]
        fn game_result_filled() {
            let mut tic = TicTacToe::start();

            tic.board = [Some(Player::Max); 9];
            assert_eq!(tic.result(), Some(WinX.score()));

            tic.board = [Some(Player::Min); 9];
            assert_eq!(tic.result(), Some(WinO.score()));
        }

        #[test]
        fn game_result_horizotal() {
            let mut tic = TicTacToe::start();
            tic.board = [O, O, O, X, X, O, X, O, X];

            assert_eq!(tic.result(), Some(WinO.score()));

            tic.board = [O, X, O, X, O, O, X, X, X];
            assert_eq!(tic.result(), Some(WinX.score()));
        }

        #[test]
        fn game_result_virtical() {
            let mut tic = TicTacToe::start();
            tic.board = [O, X, O, O, X, X, O, O, X];
            assert_eq!(tic.result(), Some(WinO.score()));
        }

        #[test]
        fn game_result_diagonal() {
            let mut tic = TicTacToe::start();
            tic.board = [X, O, O, O, X, O, X, O, X];

            assert_eq!(tic.result(), Some(WinX.score()));
        }

        #[test]
        fn game_result_draw() {
            let mut tic = TicTacToe::start();
            tic.board = [X, O, X, O, X, O, O, X, O];

            assert_eq!(tic.result(), Some(Draw.score()));
        }
    }

    #[test]
    fn test_minimax() {
        use TicTacToeSquare::*;

        let mut tic = TicTacToe::start();
        assert_eq!(minimax(&mut tic), 0); // tictactoe always results in a draw

        tic.make_move_mut(B2);
        tic.make_move_mut(B3);
        assert_eq!(minimax(&mut tic), 1); // forced win for white
    }
}
