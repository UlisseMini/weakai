pub mod tictactoe;

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::time::{Duration, Instant};

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
    fn make_move(&self, mv: &Self::Move) -> Self;

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
        let mv_score = minimax(&board.make_move(&mv));
        score = player.best(score, mv_score);
    }

    return score;
}

/// An MCTS node
// TODO: Move this to a mcts module to avoid name clobbering
struct Node<T: BoardGame> {
    t: i16, // total reward
    n: i16, // visit count
    board: T,

    children: Vec<usize>,
    parent: usize,
}

impl<T: BoardGame> Node<T> {
    fn new(board: T) -> Self {
        Self {
            t: 0,
            n: 0,
            board: board,
            children: vec![],
            parent: 0, // temporary, caller will overwrite this.
        }
    }

    fn avg_value(&self) -> f32 {
        self.t as f32 / self.n as f32
    }
}

struct Tree<T: BoardGame> {
    arena: Vec<Node<T>>,
}

impl<T: BoardGame> Tree<T> {
    fn new() -> Self {
        Self { arena: Vec::new() }
    }

    /// Get the root node of the tree. panic if no root node exists.
    fn root(&self) -> &Node<T> {
        &self.arena[0]
    }
}

impl<T: BoardGame> Deref for Tree<T> {
    type Target = Vec<Node<T>>;
    fn deref(&self) -> &Self::Target {
        &self.arena
    }
}

impl<T: BoardGame> DerefMut for Tree<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.arena
    }
}

/// Monte carlo tree search
pub struct MCTS<T: BoardGame> {
    start: Instant,
    time_limit: Duration,
    tree: Tree<T>,
}

impl<T: Clone + BoardGame> MCTS<T> {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            time_limit: Duration::new(0, 0),
            tree: Tree::new(),
        }
    }

    // TODO: Abstract quitting functionality to a function and/or thread.
    pub fn search_timed(&mut self, board: T, time_limit: Duration) -> f32 {
        self.start = Instant::now();
        self.time_limit = time_limit;
        self.search(board)
    }

    pub fn should_stop(&self) -> bool {
        self.start.elapsed() > self.time_limit
    }

    pub fn search(&mut self, board: T) -> f32 {
        self.tree.clear(); // TODO: Maintain tree between searches
        self.tree.push(Node::new(board));

        for i in 0..100_000 {
            self.search_iter(0);

            if self.should_stop() {
                break;
            }
        }

        self.tree.root().avg_value()
    }

    fn search_iter(&mut self, node: usize) {
        // if (not leaf node)
        //   search(max(ucb1(children)))
        // else
        //   if (number of simulations = 0)
        //     rollout
        //   else
        //     for each legal move
        //       add a node to the tree
        //     current = first child node (or really any)
        //     rollout
        // rollout:
        //   loop forever
        //     if end of game:
        //       return value(S)
        //   play random move

        // TODO: backprop
        // https://youtu.be/UXW2yZndl7U

        if self.tree[node].children.len() == 0 {
            // leaf node
            if self.tree[node].n == 0 {
                self.rollout(self.tree[node].board.clone());
            } else {
                //     for each legal move
                //       add a node to the tree
                //     current = first child node
                //     rollout
            }
        } else {
            // not leaf node, select highest ucb child and search them
            // TODO: Extract to helper function
            let mut best_node = self.tree[node].children[0];
            let mut best_ucb = self.ucb1(&self.tree[best_node]);
            for child in &self.tree[node].children {
                let ucb = self.ucb1(&self.tree[*child]);
                if ucb > best_ucb {
                    best_ucb = ucb;
                    best_node = *child;
                }
            }

            self.search_iter(best_node);
        }
    }

    // Play a random game, return the resulting score.
    fn rollout(&self, mut pos: T) -> i16 {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        let mut rng = thread_rng();

        while pos.result().is_none() {
            let moves = pos.legal();
            let random_move = moves[..].choose(&mut rng).unwrap();
            pos = pos.make_move(random_move);
        }

        pos.result().unwrap()
    }

    fn ucb1(&self, node: &Node<T>) -> f32 {
        let (t, n) = (node.t as f32, node.n as f32);
        let parent_visits = self.tree.root().n as f32;

        // this gives us a measure of how much our parent loves us.
        // we take the natural logarithm to scale it down of course.
        let exploration = 2.0 * f32::sqrt(f32::ln(parent_visits) / n);

        let exploitation = if node.n == 0 {
            f32::MAX - exploration
        } else {
            t / n
        };
        return exploitation + exploration;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcts_tictactoe() {
        use tictactoe::*;
        use TicTacToeSquare::*;
        let thinking_time = Duration::from_millis(100);

        let mut mcts: MCTS<TicTacToe> = MCTS::new();
        let tic = TicTacToe::start();
        let score = mcts.search_timed(tic.clone(), thinking_time);
        eprintln!("score {} want 0\n{}\n", score, tic);

        let tic = tic.make_move(&B2);
        let tic = tic.make_move(&B3);

        let score = mcts.search_timed(tic.clone(), thinking_time);
        eprintln!("score {} want 1\n{}\n", score, tic);
        unimplemented!();
    }

    #[test]
    fn minimax_tictactoe() {
        use tictactoe::*;
        use TicTacToeSquare::*;

        let tic = TicTacToe::start();
        assert_eq!(minimax(&tic), 0); // tictactoe always results in a draw

        let tic = tic.make_move(&B2);
        let tic = tic.make_move(&B3);
        assert!(minimax(&tic) > 0); // forced win for white
    }

    #[test]
    fn mcts_rollout_tictactoe() {
        // TODO: random seed to make test deterministic

        use tictactoe::*;

        let tic = TicTacToe::start();

        let mut mcts: MCTS<TicTacToe> = MCTS::new();
        let mut t = 0;
        let n = 1000;
        for i in 0..n {
            t += mcts.rollout(tic.clone());
        }

        let avg = (t as f32) / (n as f32);
        assert!(avg < 0.9 && avg > 0.1);
    }
}
