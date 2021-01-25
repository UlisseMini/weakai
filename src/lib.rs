pub mod tictactoe;
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

    /// Make a legal move, and return a new board
    fn make_move(&self, mv: Self::Move) -> Self;

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
pub fn minimax<T>(board: &T) -> i16
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
        let mv_score = minimax(&board.make_move(mv));
        score = player.best(score, mv_score);
    }

    return score;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_tictactoe() {
        use tictactoe::*;
        use TicTacToeSquare::*;

        let tic = TicTacToe::start();
        assert_eq!(minimax(&tic), 0); // tictactoe always results in a draw

        let tic = tic.make_move(B2);
        let tic = tic.make_move(B3);
        assert!(minimax(&tic) > 0); // forced win for white
    }
}
