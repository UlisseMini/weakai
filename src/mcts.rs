use crate::BoardGame;
use rand::prelude::*;
use rand::rngs::StdRng;

#[derive(Debug)]
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
    rng: StdRng,
}

impl<T: BoardGame> GameTree<T> {
    pub fn new(root: T) -> Self {
        Self {
            root: 0,
            arena: vec![Node::new(root, 0)],
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Choose the best move for a board
    pub fn choose(&self, pos: &T) -> (T::Move, i16) {
        let mut legal = pos.legal();
        (legal.remove(0), 0)
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
        self.arena[node.parent].children.push(node_idx);
        self.arena.push(node);
        node_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tictactoe::TicTacToe;

    #[test]
    fn tictactoe() {
        // let tic = TicTacToe::start();
        // let mut gt = GameTree::new(tic.clone());

        // for i in 0..10000 {
        //     gt.step();
        //     // eprintln!("{} / {}", gt.arena[0].score, gt.arena[0].visits);
        // }
        // panic!("{:#?}", gt.arena[0]);
    }

    #[test]
    fn mcts_vs_alphabeta_tictactoe() {
        use crate::{alphabeta_mv, Player};

        let mut tic = TicTacToe::start();

        // "train" the mcts tree/agent
        let mut mcts_tree = GameTree::new(tic.clone());
        for _ in 0..1000 {
            mcts_tree.step();
        }

        // play a game from the starting position
        while tic.result().is_none() {
            let (mv, eval) = match tic.turn() {
                Player::Max => alphabeta_mv(&tic),
                // Player::Min => alphabeta_mv(&tic),
                Player::Min => mcts_tree.choose(&tic),
            };
            tic = tic.make_move(&mv);

            eprintln!("{}eval: {}\n", tic, eval);
        }

        let result = tic.result().unwrap();
        eprintln!("\nresult: {}\nscore: {}", tic, result);
        assert_eq!(result, 0);
    }
}
