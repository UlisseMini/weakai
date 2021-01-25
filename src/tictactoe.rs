use crate::*;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TicTacToe {
    turn: Player,               // The player to move, X=MAX, O=MIN
    board: [Option<Player>; 9], // The board, from the top left down
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
            board: [None; 9],
        }
    }

    fn turn(&self) -> Player {
        self.turn
    }

    fn make_move(&self, mv: Self::Move) -> Self {
        let mut clone = self.clone();
        clone.board[mv as usize] = Some(self.turn);
        clone.turn = self.turn.other();
        clone
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
        let tic = TicTacToe::start();
        assert_eq!(tic.legal(), vec![A3, B3, C3, A2, B2, C2, A1, B1, C1]);

        let tic = tic.make_move(A3);
        eprintln!("after A3\n{}", tic);
        assert_eq!(tic.legal(), vec![B3, C3, A2, B2, C2, A1, B1, C1]);
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
