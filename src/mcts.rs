use crate::BoardGame;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Node<T> {
    pos: T,
    visits: u64,
    score: i64, // total score
    parent: usize,
    children: Vec<usize>,
}

impl<T> Node<T> {
    fn new(pos: T, parent: usize) -> Self {
        Self {
            pos: pos,
            visits: 0,
            score: 0,
            parent: parent,
            children: Vec::new(),
        }
    }
}

pub struct GameTree<T: BoardGame> {
    root: usize,
    arena: Vec<Node<T>>,
    positions: HashMap<T, usize>,
    rng: StdRng,
}

impl<T: BoardGame> GameTree<T> {
    pub fn new(root: T) -> Self {
        let root_node = Node::new(root, 0);
        let mut tree = Self {
            root: 0,
            arena: vec![root_node.clone()],
            rng: StdRng::seed_from_u64(42),
            positions: HashMap::new(),
        };

        tree.push_node(root_node);
        tree
    }

    /// Choose the best move for a board
    pub fn choose(&self, pos: &T) -> (T::Move, f32) {
        let legal = pos.legal();

        legal
            .into_iter()
            .map(|mv| {
                let new_pos = pos.make_move(&mv);
                let node_idx = self.positions.get(&new_pos).unwrap();
                let node = &self.arena[*node_idx];
                let score = if node.visits != 0 {
                    node.score as f32 / node.visits as f32
                } else {
                    -f32::INFINITY
                };

                (mv, score * pos.turn().factor() as f32)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }

    /// Run a single iteration of mcts, and expand the game tree.
    pub fn step(&mut self) {
        let node_idx = self.select();
        self.expand(node_idx);
        let reward = self.simulate(node_idx);
        self.backprop(node_idx, reward);
    }

    fn select(&self) -> usize {
        let mut node = self.root;

        while !self.leaf(node) {
            node = self.select_best_child(node);
        }

        node
    }

    fn select_best_child(&self, node: usize) -> usize {
        let children = &self.arena[node].children;

        // TODO: python can do max(children, key=selfpromise), can rust?
        let mut best = children[0];
        let mut best_promise = self.promise(best);

        for child in children[1..].iter() {
            let child = *child;
            let promise = self.promise(child);
            if promise > best_promise {
                best_promise = promise;
                best = child;
            }
        }

        // let promises: Vec<f32> = children.iter().map(|c| self.promise(*c)).collect();
        // eprintln!(
        //     "select promise: {} of {:?}\n{}",
        //     best_promise, promises, self.arena[best].pos
        // );
        best
    }

    fn promise(&self, node: usize) -> f32 {
        let node = &self.arena[node];

        if node.visits == 0 {
            // you must visit unvisited nodes, period
            return f32::INFINITY;
        }

        let parent = &self.arena[node.parent];
        let avg_score = node.score as f32 / node.visits as f32;

        let exploration = 2.0 * f32::sqrt(f32::ln(parent.visits as f32) / node.visits as f32);

        let factor = node.pos.turn().other().factor() as f32;
        return factor * avg_score + exploration;
    }

    fn expand(&mut self, node_idx: usize) {
        if let Some(_result) = self.arena[node_idx].pos.result() {
            // I think we can just return, nothing to expand
            return;
        }

        for mv in self.arena[node_idx].pos.legal() {
            let new_pos = self.arena[node_idx].pos.make_move(&mv);
            let new_node = Node::new(new_pos, node_idx);
            self.push_node(new_node);
        }
    }

    fn simulate(&mut self, node_idx: usize) -> i16 {
        let mut pos = self.arena[node_idx].pos.clone();

        loop {
            match pos.result() {
                Some(result) => return result,

                None => {
                    let legal = pos.legal();
                    // result is None, so moves cannot be []
                    let mv = legal.choose(&mut self.rng).unwrap();
                    pos = pos.make_move(mv);
                }
            }
        }
    }

    fn backprop(&mut self, mut node_idx: usize, reward: i16) {
        loop {
            self.arena[node_idx].score += reward as i64;
            self.arena[node_idx].visits += 1;

            let parent = self.arena[node_idx].parent;
            if parent == node_idx {
                break;
            }
            node_idx = parent;
            // reward = -reward; TODO: invert? I want abs score tho
        }
    }

    fn leaf(&self, node: usize) -> bool {
        self.arena[node].children.len() == 0
    }

    fn push_node(&mut self, node: Node<T>) -> usize {
        let node_idx = self.arena.len();

        self.positions.insert(node.pos.clone(), node_idx);

        self.arena[node.parent].children.push(node_idx);
        self.arena.push(node);

        node_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tictactoe::TicTacToe;

    // TODO: Play as both sides
    fn mcts_vs_alphabeta<T: BoardGame>(start: T, want: i16) {
        use crate::{alphabeta_mv, Player};

        let mcts_ai = |pos: &T| {
            let mut mcts_tree = GameTree::new(pos.clone());
            for _ in 0..1000 {
                mcts_tree.step();
            }
            mcts_tree.choose(pos)
        };

        let alphabeta_ai = |pos: &T| {
            let (mv, eval) = alphabeta_mv(pos);
            (mv, eval as f32)
        };

        // play a game from the starting position using max_ai and min_ai
        let play_game =
            |start: &T, max_ai: fn(&T) -> (T::Move, f32), min_ai: fn(&T) -> (T::Move, f32)| {
                let mut pos = start.clone();

                while pos.result().is_none() {
                    let (mv, eval) = match pos.turn() {
                        Player::Max => max_ai(&pos),
                        Player::Min => min_ai(&pos),
                    };
                    pos = pos.make_move(&mv);

                    eprintln!("{}eval: {}\n", pos, eval);
                }
                (pos.result().unwrap(), pos)
            };

        eprintln!("alphabeta vs mcts");
        let (result, end_pos) = play_game(&start, alphabeta_ai, mcts_ai);
        eprintln!("\nresult: {} score: {} want {}", end_pos, result, want);
        assert_eq!(result, want);

        eprintln!("mcts vs alphabeta");
        let (result, end_pos) = play_game(&start, mcts_ai, alphabeta_ai);
        eprintln!("\nresult: {} score: {} want {}", end_pos, result, want);
        assert_eq!(result, want);
    }

    #[test]
    fn mcts_vs_alphabeta_tictactoe_startpos() {
        mcts_vs_alphabeta(TicTacToe::start(), 0);
    }

    #[test]
    fn mcts_vs_alphabeta_tictactoe_whitewin() {
        use crate::tictactoe::*;
        use TicTacToeSquare::*;

        mcts_vs_alphabeta(TicTacToe::start().make_move(&B2).make_move(&B3), 1);
    }

    #[test]
    fn mcts_vs_alphabeta_tictactoe_blackwin() {
        use crate::tictactoe::*;
        use TicTacToeSquare::*;
        let tic = TicTacToe::start()
            .make_move(&B2)
            .make_move(&B3)
            .make_move(&B1)
            .make_move(&C3)
            .make_move(&A1);

        // yes black wins in one, its hard to find a position where black forces
        // a win ok

        mcts_vs_alphabeta(tic, -1);
    }
}
