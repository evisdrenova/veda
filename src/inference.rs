use crate::loader::{ModelConfig, ModelLoader, Tokenizer};
use std::collections::HashMap;

// we could  borrow referencs to the loader but that might make things more complicated
pub struct InferenceEngine {
    loader: ModelLoader,
}

impl InferenceEngine {
    pub fn new(loader: ModelLoader) -> Self {
        Self { loader }
    }

    fn config(&self) -> &ModelConfig {
        self.loader.config()
    }

    fn tokenizer(&self) -> &Tokenizer {
        self.loader.tokenizer()
    }
}
