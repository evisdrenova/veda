use crate::loader::{ModelConfig, ModelLoader, QuantType, TensorView, Tokenizer};
use std::collections::HashMap;

// we could  borrow referencs to the loader but that might make things more complicated
pub struct InferenceEngine {
    loader: ModelLoader,
}
#[derive(Debug)]
pub enum InferenceError {
    TensorNotFound(String),
    InvalidToken(u32, usize),
    SequenceTooLong(usize, usize),
    UnsupportedQuantization(QuantType),
    ShapeMismatch(String),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            InferenceError::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            InferenceError::InvalidToken(token, vocab_size) => {
                write!(f, "Invalid token {} (vocab size: {})", token, vocab_size)
            }
            InferenceError::SequenceTooLong(len, max) => {
                write!(f, "Sequence too long: {} > {}", len, max)
            }
            InferenceError::UnsupportedQuantization(qt) => {
                write!(f, "Unsupported quantization: {:?}", qt)
            }
            InferenceError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
        }
    }
}

impl std::error::Error for InferenceError {}

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

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String, InferenceError> {
        println!("🔄 Starting text generation...");

        // Tokenize input
        let start = std::time::Instant::now();
        let mut tokens = self.tokenizer().encode(prompt);

        // Add BOS token if needed
        if self.tokenizer().add_bos_token
            && !tokens.is_empty()
            && tokens[0] != self.tokenizer().bos_token_id
        {
            tokens.insert(0, self.tokenizer().bos_token_id);
        }

        println!("⏱️  Tokenization: {:?}", start.elapsed());
        println!(
            "📝 Input tokens: {:?} ({})",
            &tokens[..tokens.len().min(10)],
            tokens.len()
        );

        // Generate tokens one by one
        for i in 0..max_tokens {
            let token_start = std::time::Instant::now();

            // Forward pass through the model
            let logits = self.forward(&tokens)?;

            // Sample next token (greedy for now)
            let next_token = self.sample_token(&logits);
            tokens.push(next_token);

            println!(
                "🔢 Token {}: {} ({:?})",
                i + 1,
                next_token,
                token_start.elapsed()
            );

            // Stop on EOS token
            if next_token == self.tokenizer().eos_token_id {
                break;
            }
        }

        // Decode result
        let result = self.tokenizer().decode(&tokens);
        Ok(result)
    }

    fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let forward_start = std::time::Instant::now();

        let seq_len = tokens.len();
        if seq_len > self.config().context_length as usize {
            return Err(InferenceError::SequenceTooLong(
                seq_len,
                self.config().context_length as usize,
            ));
        }

        // Token embeddings
        let emb_start = std::time::Instant::now();
        let mut hidden_states = self.token_embeddings(tokens)?;
        println!("    📚 Token embeddings: {:?}", emb_start.elapsed());

        // Position embeddings
        let pos_start = std::time::Instant::now();
        self.add_position_embeddings(&mut hidden_states, seq_len)?;
        println!("    📍 Position embeddings: {:?}", pos_start.elapsed());

        // Transformer blocks
        let blocks_start = std::time::Instant::now();
        for layer in 0..self.config().block_count {
            let layer_start = std::time::Instant::now();
            hidden_states = self.transformer_block(hidden_states, layer as usize)?;
            println!("    🧠 Layer {}: {:?}", layer, layer_start.elapsed());
        }
        println!("    🔄 All layers: {:?}", blocks_start.elapsed());

        // Final layer norm
        let norm_start = std::time::Instant::now();
        hidden_states = self.layer_norm(&hidden_states, "norm")?;
        println!("    📏 Final norm: {:?}", norm_start.elapsed());

        // Output projection
        let output_start = std::time::Instant::now();
        let logits = self.output_projection(&hidden_states)?;
        println!("    🎯 Output projection: {:?}", output_start.elapsed());

        println!("  🧠 Total forward pass: {:?}", forward_start.elapsed());

        // Return logits for the last token
        let vocab_size = self.config().vocab_size as usize;
        let last_token_logits = logits[(seq_len - 1) * vocab_size..seq_len * vocab_size].to_vec();

        Ok(last_token_logits)
    }
    fn token_embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let token_emb = self
            .get_tensor("token_emb.weight")
            .or_else(|| self.get_tensor("tok_embeddings.weight"))
            .or_else(|| self.get_tensor("embed_tokens.weight"))
            .ok_or(InferenceError::TensorNotFound(
                "token embedding".to_string(),
            ))?;

        let vocab_size = token_emb.shape()[0] as usize;
        let embed_dim = token_emb.shape()[1] as usize;

        // Convert tensor data to f32 (assuming F32 for now)
        let weights = self.tensor_to_f32(&token_emb)?;

        // Look up embeddings for each token
        let mut embeddings = Vec::with_capacity(tokens.len() * embed_dim);
        for &token_id in tokens {
            let token_idx = token_id as usize;
            if token_idx >= vocab_size {
                return Err(InferenceError::InvalidToken(token_id, vocab_size));
            }

            let start_idx = token_idx * embed_dim;
            let end_idx = start_idx + embed_dim;
            embeddings.extend_from_slice(&weights[start_idx..end_idx]);
        }

        Ok(embeddings)
    }

    /// Add position embeddings
    fn add_position_embeddings(
        &self,
        hidden_states: &mut [f32],
        seq_len: usize,
    ) -> Result<(), InferenceError> {
        let pos_emb = self
            .get_tensor("pos_emb.weight")
            .or_else(|| self.get_tensor("pos_embeddings.weight"))
            .ok_or(InferenceError::TensorNotFound(
                "position embedding".to_string(),
            ))?;

        let embed_dim = pos_emb.shape()[1] as usize;
        let pos_weights = self.tensor_to_f32(&pos_emb)?;

        // Add position embeddings to each token
        for (pos, chunk) in hidden_states.chunks_mut(embed_dim).enumerate() {
            if pos >= seq_len {
                break;
            }

            let pos_start = pos * embed_dim;
            let pos_end = pos_start + embed_dim;

            for (i, &pos_val) in pos_weights[pos_start..pos_end].iter().enumerate() {
                chunk[i] += pos_val;
            }
        }

        Ok(())
    }

    /// Single transformer block
    fn transformer_block(
        &self,
        mut hidden_states: Vec<f32>,
        layer: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let seq_len = hidden_states.len() / self.config().embedding_length as usize;
        let embed_dim = self.config().embedding_length as usize;

        // 1. Pre-attention layer norm
        let normed_input = self.layer_norm(&hidden_states, &format!("blk.{}.attn_norm", layer))?;

        // 2. Self-attention
        let attn_output = self.self_attention(&normed_input, layer)?;

        // 3. Residual connection
        for i in 0..hidden_states.len() {
            hidden_states[i] += attn_output[i];
        }

        // 4. Pre-FFN layer norm
        let normed_hidden = self.layer_norm(&hidden_states, &format!("blk.{}.ffn_norm", layer))?;

        // 5. Feed-forward network
        let ffn_output = self.feed_forward(&normed_hidden, layer)?;

        // 6. Residual connection
        for i in 0..hidden_states.len() {
            hidden_states[i] += ffn_output[i];
        }

        Ok(hidden_states)
    }

    /// Self-attention mechanism (simplified)
    fn self_attention(
        &self,
        hidden_states: &[f32],
        layer: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let seq_len = hidden_states.len() / self.config().embedding_length as usize;
        let embed_dim = self.config().embedding_length as usize;
        let num_heads = self.config().attention_head_count as usize;
        let head_dim = embed_dim / num_heads;

        // Get attention weights
        let wq = self
            .get_tensor(&format!("blk.{}.attn_q.weight", layer))
            .ok_or_else(|| {
                InferenceError::TensorNotFound(format!("layer {} query weights", layer))
            })?;
        let wk = self
            .get_tensor(&format!("blk.{}.attn_k.weight", layer))
            .ok_or_else(|| {
                InferenceError::TensorNotFound(format!("layer {} key weights", layer))
            })?;
        let wv = self
            .get_tensor(&format!("blk.{}.attn_v.weight", layer))
            .ok_or_else(|| {
                InferenceError::TensorNotFound(format!("layer {} value weights", layer))
            })?;
        let wo = self
            .get_tensor(&format!("blk.{}.attn_output.weight", layer))
            .ok_or_else(|| {
                InferenceError::TensorNotFound(format!("layer {} output weights", layer))
            })?;

        // Convert to f32
        let wq_f32 = self.tensor_to_f32(&wq)?;
        let wk_f32 = self.tensor_to_f32(&wk)?;
        let wv_f32 = self.tensor_to_f32(&wv)?;
        let wo_f32 = self.tensor_to_f32(&wo)?;

        // Simplified attention (single-head for now)
        // This is a basic implementation - real version would need:
        // - Multi-head attention
        // - Causal masking
        // - RoPE position encoding
        // - KV caching for efficiency

        let mut output = vec![0.0; hidden_states.len()];

        // For now, just do a simple linear transformation as placeholder
        // TODO: Implement proper multi-head attention
        self.linear_transform(hidden_states, &wo_f32, &mut output, embed_dim, embed_dim)?;

        Ok(output)
    }

    /// Feed-forward network
    fn feed_forward(
        &self,
        hidden_states: &[f32],
        layer: usize,
    ) -> Result<Vec<f32>, InferenceError> {
        let embed_dim = self.config().embedding_length as usize;
        let ff_dim = self.config().feed_forward_length as usize;

        // Get FFN weights
        let w1 = self
            .get_tensor(&format!("blk.{}.ffn_up.weight", layer))
            .or_else(|| self.get_tensor(&format!("blk.{}.ffn_gate.weight", layer)))
            .ok_or_else(|| {
                InferenceError::TensorNotFound(format!("layer {} FFN up weights", layer))
            })?;
        let w2 = self
            .get_tensor(&format!("blk.{}.ffn_down.weight", layer))
            .ok_or_else(|| {
                InferenceError::TensorNotFound(format!("layer {} FFN down weights", layer))
            })?;

        let w1_f32 = self.tensor_to_f32(&w1)?;
        let w2_f32 = self.tensor_to_f32(&w2)?;

        // Up projection
        let mut intermediate = vec![0.0; hidden_states.len() / embed_dim * ff_dim];
        self.linear_transform(hidden_states, &w1_f32, &mut intermediate, embed_dim, ff_dim)?;

        // Activation (GeLU/SiLU)
        for x in &mut intermediate {
            *x = gelu(*x); // or silu(*x) depending on model
        }

        // Down projection
        let mut output = vec![0.0; hidden_states.len()];
        self.linear_transform(&intermediate, &w2_f32, &mut output, ff_dim, embed_dim)?;

        Ok(output)
    }

    /// Layer normalization
    fn layer_norm(
        &self,
        hidden_states: &[f32],
        weight_name: &str,
    ) -> Result<Vec<f32>, InferenceError> {
        let norm_weight = self
            .get_tensor(&format!("{}.weight", weight_name))
            .ok_or_else(|| {
                InferenceError::TensorNotFound(format!("{} norm weights", weight_name))
            })?;

        let weights = self.tensor_to_f32(&norm_weight)?;
        let embed_dim = weights.len();
        let seq_len = hidden_states.len() / embed_dim;

        let mut output = vec![0.0; hidden_states.len()];
        let eps = self.config().attention_layer_norm_rms_epsilon;

        for seq_idx in 0..seq_len {
            let start = seq_idx * embed_dim;
            let end = start + embed_dim;
            let input_slice = &hidden_states[start..end];
            let output_slice = &mut output[start..end];

            // RMS norm
            let mean_square: f32 =
                input_slice.iter().map(|x| x * x).sum::<f32>() / embed_dim as f32;
            let rms = (mean_square + eps).sqrt();

            for i in 0..embed_dim {
                output_slice[i] = input_slice[i] / rms * weights[i];
            }
        }

        Ok(output)
    }

    /// Output projection (language modeling head)
    fn output_projection(&self, hidden_states: &[f32]) -> Result<Vec<f32>, InferenceError> {
        // Use the same embedding weights (tied weights)
        let token_emb = self
            .get_tensor("token_emb.weight")
            .or_else(|| self.get_tensor("tok_embeddings.weight"))
            .or_else(|| self.get_tensor("embed_tokens.weight"))
            .ok_or(InferenceError::TensorNotFound(
                "token embedding for output".to_string(),
            ))?;

        let weights = self.tensor_to_f32(&token_emb)?;
        let embed_dim = self.config().embedding_length as usize;
        let vocab_size = self.config().vocab_size as usize;
        let seq_len = hidden_states.len() / embed_dim;

        let mut logits = vec![0.0; seq_len * vocab_size];

        // Matrix multiply: [seq_len, embed_dim] @ [vocab_size, embed_dim]^T = [seq_len, vocab_size]
        for seq_idx in 0..seq_len {
            let input_start = seq_idx * embed_dim;
            let output_start = seq_idx * vocab_size;

            for vocab_idx in 0..vocab_size {
                let weight_start = vocab_idx * embed_dim;
                let mut sum = 0.0;

                for dim in 0..embed_dim {
                    sum += hidden_states[input_start + dim] * weights[weight_start + dim];
                }

                logits[output_start + vocab_idx] = sum;
            }
        }

        Ok(logits)
    }

    /// Sample next token from logits
    fn sample_token(&self, logits: &[f32]) -> u32 {
        // Greedy sampling (pick highest logit)
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        max_idx as u32
    }

    // Helper methods
    fn get_tensor(&self, name: &str) -> Option<TensorView> {
        self.loader.get_tensor(name)
    }

    fn tensor_to_f32(&self, tensor: &TensorView) -> Result<Vec<f32>, InferenceError> {
        match tensor.quant_type() {
            QuantType::F32 => {
                let data = tensor.data;
                let mut result = Vec::with_capacity(data.len() / 4);

                for chunk in data.chunks_exact(4) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    result.push(value);
                }

                Ok(result)
            }
            QuantType::F16 => {
                // TODO: Implement F16 dequantization
                Err(InferenceError::UnsupportedQuantization(tensor.quant_type()))
            }
            _ => {
                // TODO: Implement quantized formats (Q4_0, Q8_0, etc.)
                Err(InferenceError::UnsupportedQuantization(tensor.quant_type()))
            }
        }
    }

    fn linear_transform(
        &self,
        input: &[f32],
        weights: &[f32],
        output: &mut [f32],
        input_dim: usize,
        output_dim: usize,
    ) -> Result<(), InferenceError> {
        let seq_len = input.len() / input_dim;

        for seq_idx in 0..seq_len {
            let input_start = seq_idx * input_dim;
            let output_start = seq_idx * output_dim;

            for out_idx in 0..output_dim {
                let weight_start = out_idx * input_dim;
                let mut sum = 0.0;

                for in_idx in 0..input_dim {
                    sum += input[input_start + in_idx] * weights[weight_start + in_idx];
                }

                output[output_start + out_idx] = sum;
            }
        }

        Ok(())
    }
}

// Activation functions
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
