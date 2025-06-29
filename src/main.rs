use std::time::Instant;

mod loader;

use crate::loader::GgufLoader;

fn main() -> std::io::Result<()> {
    let home = std::env::var("HOME").unwrap();

    let start = Instant::now();
    let loader = GgufLoader::load("/Users/evisdrenova/Downloads/gemma-3n-E4B-it-f16.gguf")?;
    println!("🚀 Model loaded in: {:?}", start.elapsed());

    // Inspect model configuration
    let config = loader.config();
    println!("📋 Model Config:");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Context length: {}", config.context_length);
    println!("  Embedding dims: {}", config.embedding_length);
    println!("  Layers: {}", config.block_count);
    println!("  Attention heads: {}", config.attention_head_count);

    // List some tensors
    println!("\n🎯 Available tensors:");
    for (i, name) in loader.tensor_names().enumerate() {
        if i < 10 {
            // Show first 10
            println!("  {}", name);
        }
    }
    println!(
        "  ... and {} more",
        loader.tensor_count().saturating_sub(10)
    );

    // Zero-copy tensor access
    if let Some(token_emb) = loader.get_tensor("token_emb.weight") {
        println!("\n🔍 Token embedding:");
        println!("  Shape: {:?}", token_emb.shape());
        println!("  Type: {:?}", token_emb.quant_type());
        println!("  Size: {} bytes", token_emb.data.len());
        println!("  Elements: {}", token_emb.element_count());

        // Access the raw data (zero-copy!)
        let data_slice = token_emb.data;
        println!(
            "  First 16 bytes: {:?}",
            &data_slice[..16.min(data_slice.len())]
        );
    }

    // Demonstrate fast repeated access
    let lookup_start = Instant::now();
    for _ in 0..1000 {
        let _ = loader.get_tensor("token_emb.weight");
    }
    println!("\n⚡ 1000 tensor lookups: {:?}", lookup_start.elapsed());

    Ok(())
}
