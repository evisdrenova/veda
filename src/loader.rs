use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self};
use std::path::Path;

// GGUF Data Types (same as before)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgufType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufType {
    type Error = String;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GgufType::UInt8),
            1 => Ok(GgufType::Int8),
            2 => Ok(GgufType::UInt16),
            3 => Ok(GgufType::Int16),
            4 => Ok(GgufType::UInt32),
            5 => Ok(GgufType::Int32),
            6 => Ok(GgufType::Float32),
            7 => Ok(GgufType::Bool),
            8 => Ok(GgufType::String),
            9 => Ok(GgufType::Array),
            10 => Ok(GgufType::UInt64),
            11 => Ok(GgufType::Int64),
            12 => Ok(GgufType::Float64),
            _ => Err(format!("Unknown GGUF type: {}", value)),
        }
    }
}

// Quantization formats (same as before)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
}

impl TryFrom<u32> for QuantType {
    type Error = String;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(QuantType::F32),
            1 => Ok(QuantType::F16),
            2 => Ok(QuantType::Q4_0),
            3 => Ok(QuantType::Q4_1),
            6 => Ok(QuantType::Q5_0),
            7 => Ok(QuantType::Q5_1),
            8 => Ok(QuantType::Q8_0),
            9 => Ok(QuantType::Q8_1),
            10 => Ok(QuantType::Q2_K),
            11 => Ok(QuantType::Q3_K),
            12 => Ok(QuantType::Q4_K),
            13 => Ok(QuantType::Q5_K),
            14 => Ok(QuantType::Q6_K),
            15 => Ok(QuantType::Q8_K),
            _ => Err(format!("Unknown quantization type: {}", value)),
        }
    }
}

// Tokenizer with full vocabulary support
#[derive(Debug, Default)]
pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub scores: Vec<f32>,
    pub token_types: Vec<i32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub unknown_token_id: u32,
    pub padding_token_id: u32,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub chat_template: String,
    vocab_mapping: Option<HashMap<u32, u32>>, // Maps full vocab IDs to compressed vocab IDs
    compressed_vocab_size: Option<usize>,
}

impl Tokenizer {
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn compressed_vocab_size(&self) -> usize {
        self.compressed_vocab_size.unwrap_or(self.vocab.len())
    }

    // Set the compressed vocabulary size from the embedding tensor
    pub fn set_compressed_vocab_size(&mut self, size: usize) {
        self.compressed_vocab_size = Some(size);

        // Create a mapping from the most frequent tokens to the compressed vocabulary
        // For now, we'll use a simple mapping of the first N tokens
        // In a real implementation, you'd want to map the most frequent tokens
        let mut mapping = HashMap::new();

        for i in 0..size.min(self.vocab.len()) {
            mapping.insert(i as u32, i as u32);
        }

        // Map any remaining high-frequency tokens to available slots
        self.vocab_mapping = Some(mapping);

        println!(
            "🔧 Tokenizer: Set compressed vocab size to {} (from full size {})",
            size,
            self.vocab.len()
        );
    }

    // Simple SentencePiece-like tokenization (very basic implementation)
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // This is still a simplified tokenizer
        // For production, you'd implement proper SentencePiece tokenization
        let mut tokens = Vec::new();

        // Basic word-level tokenization with subword fallback
        for word in text.split_whitespace() {
            let word_tokens = self.encode_word(word);
            tokens.extend(word_tokens);
        }

        // Map to compressed vocabulary if available
        if let Some(compressed_size) = self.compressed_vocab_size {
            tokens = self.map_to_compressed_vocab(tokens, compressed_size);
        }

        println!(
            "🔍 Debug: Encoded '{}' to {} tokens: {:?}",
            text,
            tokens.len(),
            &tokens[..tokens.len().min(10)]
        );

        tokens
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        // Try to find exact word match first
        if let Some(pos) = self.vocab.iter().position(|v| v == word) {
            return vec![pos as u32];
        }

        // Try lowercase version
        let lower_word = word.to_lowercase();
        if let Some(pos) = self.vocab.iter().position(|v| v == &lower_word) {
            return vec![pos as u32];
        }

        // Try to find partial matches (very basic subword tokenization)
        let mut result = Vec::new();
        let mut remaining = word;

        while !remaining.is_empty() {
            let mut found = false;

            // Try progressively shorter prefixes
            for len in (1..=remaining.len()).rev() {
                let prefix = &remaining[..len];
                if let Some(pos) = self.vocab.iter().position(|v| v == prefix) {
                    result.push(pos as u32);
                    remaining = &remaining[len..];
                    found = true;
                    break;
                }
            }

            if !found {
                // Fallback: use unknown token and advance by one character
                result.push(self.get_safe_unknown_token_id());
                if remaining.len() > 1 {
                    remaining = &remaining[1..];
                } else {
                    break;
                }
            }
        }

        result
    }

    fn map_to_compressed_vocab(&self, tokens: Vec<u32>, compressed_size: usize) -> Vec<u32> {
        tokens
            .into_iter()
            .map(|token| {
                if (token as usize) < compressed_size {
                    // Token is already in compressed vocabulary
                    token
                } else {
                    // Map out-of-range token to a safe alternative
                    self.get_safe_unknown_token_id()
                }
            })
            .collect()
    }

    fn get_safe_unknown_token_id(&self) -> u32 {
        let compressed_size = self.compressed_vocab_size.unwrap_or(self.vocab.len());

        // Try the configured unknown token first
        if (self.unknown_token_id as usize) < compressed_size {
            return self.unknown_token_id;
        }

        // Try other common token IDs
        let candidates = [3, 0, 1, 2]; // UNK, PAD, BOS, EOS
        for &candidate in &candidates {
            if (candidate as usize) < compressed_size {
                return candidate;
            }
        }

        // Last resort: use the last token in compressed vocabulary
        (compressed_size.saturating_sub(1)) as u32
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let compressed_size = self.compressed_vocab_size();

        tokens
            .iter()
            .filter_map(|&token_id| {
                if (token_id as usize) < compressed_size && (token_id as usize) < self.vocab.len() {
                    self.vocab.get(token_id as usize).map(|s| s.as_str())
                } else {
                    Some("<UNK>")
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    // Method to load vocabulary mapping from GGUF metadata
    pub fn load_vocab_mapping_from_metadata(&mut self, _metadata: &HashMap<String, String>) {
        // In a real implementation, you might load a vocabulary mapping
        // from the GGUF metadata if it's provided
        // For now, we'll use the simple approach above
    }
}

// Enhanced model configuration
#[derive(Debug, Default)]
pub struct ModelConfig {
    pub vocab_size: u32,
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub feed_forward_length: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub rope_freq_base: f32,
    pub rope_scaling_type: String,

    // Additional Gemma-specific parameters
    pub attention_key_length: u32,
    pub attention_value_length: u32,
    pub attention_sliding_window: u32,
    pub shared_kv_layers: u32,

    // Model metadata
    pub architecture: String,
    pub model_type: String,
    pub size_label: String,
    pub license: String,
    pub organization: String,

    // Per-layer configurations
    pub activation_sparsity_scale: Vec<f32>,
    pub sliding_window_pattern: Vec<bool>,
}

// Tensor descriptor (same as before)
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub quant_type: QuantType,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub size: u64,
}

// Zero-copy tensor view (same as before)
#[derive(Debug)]
pub struct TensorView<'a> {
    pub desc: &'a TensorDesc,
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn shape(&self) -> &[u64] {
        &self.desc.shape
    }
    pub fn quant_type(&self) -> QuantType {
        self.desc.quant_type
    }
    pub fn element_count(&self) -> u64 {
        self.desc.shape.iter().product()
    }
}

// Complete GGUF loader with tokenizer support
pub struct ModelLoader {
    _file: File,
    mmap: Mmap,
    config: ModelConfig,
    tensors: HashMap<String, TensorDesc>,
    tokenizer: Tokenizer,
}

impl ModelLoader {
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let load_start = std::time::Instant::now();

        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        println!("⏱️  File mapping: {:?}", load_start.elapsed());

        let parse_start = std::time::Instant::now();
        let mut reader = ByteReader::new(&mmap);

        // Check magic and version (same as before)
        let magic = reader.read_bytes(4)?;
        if magic != b"GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid GGUF magic number",
            ));
        }

        let version = reader.read_u32()?;
        if version != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version: {}", version),
            ));
        }

        let tensor_count = reader.read_u64()?;
        let metadata_count = reader.read_u64()?;

        println!(
            "📊 GGUF v{}: {} tensors, {} metadata entries",
            version, tensor_count, metadata_count
        );

        // Parse metadata with array support
        let mut config = ModelConfig::default();
        let mut tokenizer = Tokenizer::default();

        for i in 0..metadata_count {
            let key = reader.read_string()?;
            let value_type_raw = reader.read_u32()?;
            let value_type = GgufType::try_from(value_type_raw).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid type {} for key '{}': {}", value_type_raw, key, e),
                )
            })?;

            println!(
                "Metadata {}/{}: '{}' ({:?})",
                i + 1,
                metadata_count,
                key,
                value_type
            );

            Self::parse_metadata_value(&mut reader, &key, value_type, &mut config, &mut tokenizer)?;
        }

        println!("⏱️  Metadata parsing: {:?}", parse_start.elapsed());

        // Parse tensor directory (same as before)
        let tensor_start = std::time::Instant::now();
        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        let data_offset = reader.position();

        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dimensions = reader.read_u32()?;

            let mut shape = Vec::with_capacity(n_dimensions as usize);
            for _ in 0..n_dimensions {
                shape.push(reader.read_u64()?);
            }

            let quant_type = QuantType::try_from(reader.read_u32()?)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let offset = reader.read_u64()?;
            let element_count: u64 = shape.iter().product();
            let size = Self::calculate_tensor_size(element_count, quant_type);

            tensors.insert(
                name.clone(),
                TensorDesc {
                    name,
                    quant_type,
                    shape,
                    offset: data_offset + offset,
                    size,
                },
            );
        }

        println!("⏱️  Tensor directory: {:?}", tensor_start.elapsed());
        println!("✅ Total load time: {:?}", load_start.elapsed());

        let mut loader = ModelLoader {
            _file: file,
            mmap,
            config,
            tensors,
            tokenizer,
        };

        loader.configure_tokenizer_for_gemma3n()?;

        Ok(loader)
    }

    fn configure_tokenizer_for_gemma3n(&mut self) -> io::Result<()> {
        // Detect if this is a Gemma 3n model
        if self.config.architecture.contains("gemma3n") {
            println!("🔍 Detected Gemma 3n model, configuring vocabulary mapping...");

            // Get the actual embedding vocabulary size
            let embedding_vocab_size = self.get_embedding_vocab_size()?;

            if embedding_vocab_size < self.tokenizer.vocab_size() {
                println!(
                    "📊 Full tokenizer vocab: {} tokens",
                    self.tokenizer.vocab_size()
                );
                println!(
                    "📊 Compressed embedding vocab: {} tokens",
                    embedding_vocab_size
                );

                // Configure tokenizer for compressed vocabulary
                self.tokenizer
                    .set_compressed_vocab_size(embedding_vocab_size);

                // Update config to reflect actual vocabulary size used by model
                self.config.vocab_size = embedding_vocab_size as u32;
            }
        }

        Ok(())
    }

    fn get_embedding_vocab_size(&self) -> io::Result<usize> {
        // Try different possible embedding tensor names
        let embedding_tensors = [
            "token_embd.weight",
            "per_layer_token_embd.weight",
            "token_emb.weight",
            "embed_tokens.weight",
        ];

        for tensor_name in &embedding_tensors {
            if let Some(tensor_desc) = self.tensors.get(*tensor_name) {
                let vocab_size = tensor_desc.shape[0] as usize;
                println!(
                    "🎯 Found embedding tensor '{}' with vocab size: {}",
                    tensor_name, vocab_size
                );
                return Ok(vocab_size);
            }
        }

        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No embedding tensor found to determine vocabulary size",
        ))
    }

    // Helper method for inference engine
    pub fn get_effective_vocab_size(&self) -> usize {
        self.tokenizer.compressed_vocab_size()
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn get_tensor(&self, name: &str) -> Option<TensorView> {
        let desc = self.tensors.get(name)?;
        let start = desc.offset as usize;
        let end = start + desc.size as usize;

        if end > self.mmap.len() {
            return None;
        }

        Some(TensorView {
            desc,
            data: &self.mmap[start..end],
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    // Enhanced metadata parsing with array support
    fn parse_metadata_value(
        reader: &mut ByteReader,
        key: &str,
        value_type: GgufType,
        config: &mut ModelConfig,
        tokenizer: &mut Tokenizer,
    ) -> io::Result<()> {
        match value_type {
            GgufType::UInt32 => {
                let value = reader.read_u32()?;
                println!("  {} = {}", key, value);

                // Model-agnostic parameter matching
                if key.ends_with(".context_length") {
                    config.context_length = value;
                } else if key.ends_with(".embedding_length") {
                    config.embedding_length = value;
                } else if key.ends_with(".block_count") {
                    config.block_count = value;
                } else if key.ends_with(".feed_forward_length") {
                    config.feed_forward_length = value;
                } else if key.ends_with(".attention.head_count") {
                    config.attention_head_count = value;
                } else if key.ends_with(".attention.head_count_kv") {
                    config.attention_head_count_kv = value;
                } else if key.ends_with(".attention.key_length") {
                    config.attention_key_length = value;
                } else if key.ends_with(".attention.value_length") {
                    config.attention_value_length = value;
                } else if key.ends_with(".attention.sliding_window") {
                    config.attention_sliding_window = value;
                } else if key.ends_with(".attention.shared_kv_layers") {
                    config.shared_kv_layers = value;
                } else if key == "tokenizer.ggml.bos_token_id" {
                    tokenizer.bos_token_id = value;
                } else if key == "tokenizer.ggml.eos_token_id" {
                    tokenizer.eos_token_id = value;
                } else if key == "tokenizer.ggml.unknown_token_id" {
                    tokenizer.unknown_token_id = value;
                } else if key == "tokenizer.ggml.padding_token_id" {
                    tokenizer.padding_token_id = value;
                }
            }
            GgufType::Float32 => {
                let value = reader.read_f32()?;
                println!("  {} = {}", key, value);
                if key.ends_with(".attention.layer_norm_rms_epsilon") {
                    config.attention_layer_norm_rms_epsilon = value;
                } else if key.ends_with(".rope.freq_base") {
                    config.rope_freq_base = value;
                }
            }
            GgufType::String => {
                let value = reader.read_string()?;
                let display_value = if value.len() > 100 {
                    format!("{}...", &value[..100])
                } else {
                    value.clone()
                };
                println!("  {} = '{}'", key, display_value);
                if key == "general.architecture" {
                    config.architecture = value;
                } else if key == "general.type" {
                    config.model_type = value;
                } else if key == "general.size_label" {
                    config.size_label = value;
                } else if key == "general.license" {
                    config.license = value;
                } else if key.ends_with(".organization") {
                    config.organization = value;
                } else if key == "tokenizer.chat_template" {
                    tokenizer.chat_template = value;
                }
            }
            GgufType::Bool => {
                let value = reader.read_bytes(1)?[0] != 0;
                println!("  {} = {}", key, value);
                if key == "tokenizer.ggml.add_bos_token" {
                    tokenizer.add_bos_token = value;
                } else if key == "tokenizer.ggml.add_eos_token" {
                    tokenizer.add_eos_token = value;
                }
            }
            GgufType::Array => {
                let array_type_raw = reader.read_u32()?;
                let array_type = GgufType::try_from(array_type_raw)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let array_len = reader.read_u64()?;

                println!("  {} = <Array: {:?}[{}]>", key, array_type, array_len);

                // Parse important arrays
                match key {
                    "tokenizer.ggml.tokens" => {
                        println!("    📚 Loading vocabulary...");
                        tokenizer.vocab = Self::parse_string_array(reader, array_len)?;
                        config.vocab_size = array_len as u32;
                    }
                    "tokenizer.ggml.scores" => {
                        println!("    📊 Loading token scores...");
                        tokenizer.scores = Self::parse_float_array(reader, array_len)?;
                    }
                    "tokenizer.ggml.token_type" => {
                        println!("    🏷️  Loading token types...");
                        tokenizer.token_types = Self::parse_int_array(reader, array_len)?;
                    }
                    key if key.ends_with(".activation_sparsity_scale") => {
                        println!("    ⚡ Loading sparsity scales...");
                        config.activation_sparsity_scale =
                            Self::parse_float_array(reader, array_len)?;
                    }
                    key if key.ends_with(".attention.sliding_window_pattern") => {
                        println!("    🪟 Loading sliding window patterns...");
                        config.sliding_window_pattern = Self::parse_bool_array(reader, array_len)?;
                    }
                    _ => {
                        // Skip other arrays
                        Self::skip_array(reader, array_type, array_len)?;
                    }
                }
            }
            _ => {
                println!("  {} = <{:?} - skipped>", key, value_type);
                reader.skip_value(value_type)?;
            }
        }
        Ok(())
    }

    // Array parsing methods
    fn parse_string_array(reader: &mut ByteReader, len: u64) -> io::Result<Vec<String>> {
        let mut result = Vec::with_capacity(len as usize);
        for _ in 0..len {
            result.push(reader.read_string()?);
        }
        Ok(result)
    }

    fn parse_float_array(reader: &mut ByteReader, len: u64) -> io::Result<Vec<f32>> {
        let mut result = Vec::with_capacity(len as usize);
        for _ in 0..len {
            result.push(reader.read_f32()?);
        }
        Ok(result)
    }

    fn parse_int_array(reader: &mut ByteReader, len: u64) -> io::Result<Vec<i32>> {
        let mut result = Vec::with_capacity(len as usize);
        for _ in 0..len {
            result.push(reader.read_i32()?);
        }
        Ok(result)
    }

    fn parse_bool_array(reader: &mut ByteReader, len: u64) -> io::Result<Vec<bool>> {
        let mut result = Vec::with_capacity(len as usize);
        for _ in 0..len {
            result.push(reader.read_bytes(1)?[0] != 0);
        }
        Ok(result)
    }

    fn skip_array(reader: &mut ByteReader, array_type: GgufType, len: u64) -> io::Result<()> {
        for _ in 0..len {
            reader.skip_value(array_type)?;
        }
        Ok(())
    }

    fn calculate_tensor_size(element_count: u64, quant_type: QuantType) -> u64 {
        match quant_type {
            QuantType::F32 => element_count * 4,
            QuantType::F16 => element_count * 2,
            QuantType::Q4_0 => element_count / 2 + element_count / 32 * 2,
            QuantType::Q4_1 => element_count / 2 + element_count / 32 * 4,
            QuantType::Q8_0 => element_count + element_count / 32 * 2,
            QuantType::Q8_1 => element_count + element_count / 32 * 4,
            _ => element_count,
        }
    }
}

// Enhanced ByteReader with additional methods
struct ByteReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }
    fn position(&self) -> u64 {
        self.position as u64
    }

    fn read_bytes(&mut self, count: usize) -> io::Result<&'a [u8]> {
        if self.position + count > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "Unexpected EOF: trying to read {} bytes at position {}, but file is only {} bytes",
                    count,
                    self.position,
                    self.data.len()
                ),
            ));
        }
        let slice = &self.data[self.position..self.position + count];
        self.position += count;
        Ok(slice)
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i32(&mut self) -> io::Result<i32> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_string(&mut self) -> io::Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    fn skip_value(&mut self, value_type: GgufType) -> io::Result<()> {
        match value_type {
            GgufType::UInt8 | GgufType::Int8 | GgufType::Bool => {
                self.read_bytes(1)?;
            }
            GgufType::UInt16 | GgufType::Int16 => {
                self.read_bytes(2)?;
            }
            GgufType::UInt32 | GgufType::Int32 | GgufType::Float32 => {
                self.read_bytes(4)?;
            }
            GgufType::UInt64 | GgufType::Int64 | GgufType::Float64 => {
                self.read_bytes(8)?;
            }
            GgufType::String => {
                let len = self.read_u64()? as usize;
                self.read_bytes(len)?;
            }
            GgufType::Array => {
                let array_type = GgufType::try_from(self.read_u32()?)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let array_len = self.read_u64()?;
                for _ in 0..array_len {
                    self.skip_value(array_type)?;
                }
            }
        }
        Ok(())
    }
}
