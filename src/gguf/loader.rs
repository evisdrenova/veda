use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

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

// Quantization formats
#[derive(Debug, Clone, Copy, PartialEq)]
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
}

// Tensor descriptor - points to data in the mmap
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub quant_type: QuantType,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub size: u64,
}

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

pub struct GgufLoader {
    _file: File,                          // Keep file handle alive
    mmap: Mmap,                           // Memory-mapped file
    config: ModelConfig,                  // Model hyperparameters
    tensors: HashMap<String, TensorDesc>, // Tensor directory
}

impl GgufLoader {
    /// Load a GGUF model with zero-copy memory mapping
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let load_start = std::time::Instant::now();

        // Step 1: Open and memory-map the file
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        println!("⏱️  File mapping: {:?}", load_start.elapsed());

        // Step 2: Parse header and hyperparameters
        let parse_start = std::time::Instant::now();
        let mut reader = ByteReader::new(&mmap);

        // Check magic number
        let magic = reader.read_bytes(4)?;
        if magic != b"GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid GGUF magic number",
            ));
        }

        // Read version
        let version = reader.read_u32()?;
        if version != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version: {}", version),
            ));
        }

        // Read tensor count and metadata count
        let tensor_count = reader.read_u64()?;
        let metadata_count = reader.read_u64()?;

        println!(
            "📊 GGUF v{}: {} tensors, {} metadata entries",
            version, tensor_count, metadata_count
        );

        // Parse metadata (hyperparameters)
        let mut config = ModelConfig::default();
        for _ in 0..metadata_count {
            let key = reader.read_string()?;
            let value_type = GgufType::try_from(reader.read_u32()?)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            Self::parse_metadata_value(&mut reader, &key, value_type, &mut config)?;
        }

        println!("⏱️  Metadata parsing: {:?}", parse_start.elapsed());

        // Step 3: Build tensor directory
        let tensor_start = std::time::Instant::now();
        let mut tensors = HashMap::with_capacity(tensor_count as usize);

        // Parse tensor info (names, shapes, types)
        let mut data_offset = reader.position();

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

            // Calculate tensor size
            let element_count: u64 = shape.iter().product();
            let size = Self::calculate_tensor_size(element_count, quant_type);

            // insert into the hashmap
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

        Ok(GgufLoader {
            _file: file,
            mmap,
            config,
            tensors,
        })
    }
    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get tensor by name (zero-copy)
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

    /// List all tensor names
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Get tensor count
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    // Helper: Calculate tensor size based on quantization
    fn calculate_tensor_size(element_count: u64, quant_type: QuantType) -> u64 {
        match quant_type {
            QuantType::F32 => element_count * 4,
            QuantType::F16 => element_count * 2,
            QuantType::Q4_0 => element_count / 2 + element_count / 32 * 2, // 4.5 bits per element
            QuantType::Q4_1 => element_count / 2 + element_count / 32 * 4, // 4.625 bits per element
            QuantType::Q8_0 => element_count + element_count / 32 * 2,     // 8.25 bits per element
            QuantType::Q8_1 => element_count + element_count / 32 * 4,     // 8.5 bits per element
            _ => element_count, // Fallback for other types
        }
    }

    // Helper: Parse metadata values
    fn parse_metadata_value(
        reader: &mut ByteReader,
        key: &str,
        value_type: GgufType,
        config: &mut ModelConfig,
    ) -> io::Result<()> {
        match value_type {
            GgufType::UInt32 => {
                let value = reader.read_u32()?;
                match key {
                    "llama.vocab_size" => config.vocab_size = value,
                    "llama.context_length" => config.context_length = value,
                    "llama.embedding_length" => config.embedding_length = value,
                    "llama.block_count" => config.block_count = value,
                    "llama.feed_forward_length" => config.feed_forward_length = value,
                    "llama.attention.head_count" => config.attention_head_count = value,
                    "llama.attention.head_count_kv" => config.attention_head_count_kv = value,
                    _ => {} // Ignore unknown keys
                }
            }
            GgufType::Float32 => {
                let value = reader.read_f32()?;
                match key {
                    "llama.attention.layer_norm_rms_epsilon" => {
                        config.attention_layer_norm_rms_epsilon = value
                    }
                    "llama.rope.freq_base" => config.rope_freq_base = value,
                    _ => {}
                }
            }
            GgufType::String => {
                let value = reader.read_string()?;
                match key {
                    "llama.rope.scaling.type" => config.rope_scaling_type = value,
                    _ => {}
                }
            }
            // Skip other types for now
            _ => {
                reader.skip_value(value_type)?;
            }
        }
        Ok(())
    }
}

// bye reader for the mmap data
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
                "Unexpected EOF",
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
                self.position += 1;
            }
            GgufType::UInt16 | GgufType::Int16 => {
                self.position += 2;
            }
            GgufType::UInt32 | GgufType::Int32 | GgufType::Float32 => {
                self.position += 4;
            }
            GgufType::UInt64 | GgufType::Int64 | GgufType::Float64 => {
                self.position += 8;
            }
            GgufType::String => {
                let len = self.read_u64()? as usize;
                self.position += len;
            }
            GgufType::Array => {
                let _array_type = self.read_u32()?;
                let array_len = self.read_u64()?;
                // For simplicity, skip array elements (would need recursive parsing)
                for _ in 0..array_len {
                    // This is simplified - real implementation would parse based on array_type
                    self.position += 4; // Assume 4-byte elements for now
                }
            }
        }
        Ok(())
    }
}
