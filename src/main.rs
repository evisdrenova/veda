use std::io::{self, Write};
use std::time::Instant;

mod inference;
mod loader; // Your inference module

use crate::inference::InferenceEngine;
use crate::loader::ModelLoader; // Import your inference engine

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "info" => run_info(&args[2..])?,
        "generate" => run_generate(&args[2..])?,
        "chat" => run_chat(&args[2..])?,
        "benchmark" => run_benchmark(&args[2..])?,
        "tensors" => run_tensors(&args[2..])?,
        _ => print_usage(),
    }

    Ok(())
}

fn print_usage() {
    println!("🦀 GGUF CLI Tool");
    println!("");
    println!("Usage:");
    println!("  gguf-cli info <model.gguf>              Show model information");
    println!("  gguf-cli generate <model.gguf> <prompt> Generate text from prompt");
    println!("  gguf-cli chat <model.gguf>              Interactive chat mode");
    println!("  gguf-cli benchmark <model.gguf>         Benchmark loading and inference");
    println!("  gguf-cli tensors <model.gguf>           List all model tensors");
    println!("");
    println!("Examples:");
    println!("  gguf-cli info model.gguf");
    println!("  gguf-cli generate model.gguf \"Hello world\"");
    println!("  gguf-cli chat model.gguf");
    println!("  gguf-cli benchmark model.gguf");
}

fn run_info(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        println!("Error: Missing model file");
        println!("Usage: gguf-cli info <model.gguf>");
        return Ok(());
    }

    let model_path = &args[0];

    println!("🔍 Loading model: {}", model_path);
    let start = Instant::now();
    let loader = ModelLoader::load(model_path)?;
    let load_time = start.elapsed();

    println!("🚀 Model loaded in: {:?}", load_time);
    println!("{}", "=".repeat(60));

    // Model configuration
    let config = loader.config();
    println!("📋 Model Configuration:");
    println!("  Architecture: {}", config.architecture);
    println!("  Size: {}", config.size_label);
    println!("  License: {}", config.license);
    println!("  Organization: {}", config.organization);
    println!("  Vocab size:  {:?}", config.vocab_size);
    println!("  Context length:  {:?}", config.context_length);
    println!("  Embedding dims: {}", config.embedding_length);
    println!("  Layers: {}", config.block_count);
    println!("  Attention heads: {}", config.attention_head_count);
    println!("  KV heads: {}", config.attention_head_count_kv);
    println!("  Feed forward:  {:?}", config.feed_forward_length);

    // Advanced features
    if config.attention_sliding_window > 0 {
        println!("  🪟 Sliding window: {}", config.attention_sliding_window);
    }
    if config.shared_kv_layers > 0 {
        println!("  🔄 Shared KV layers: {}", config.shared_kv_layers);
    }
    if !config.activation_sparsity_scale.is_empty() {
        println!(
            "  ⚡ Sparsity layers: {}",
            config.activation_sparsity_scale.len()
        );
    }

    // Tokenizer info
    let tokenizer = loader.tokenizer();
    println!("\n🔤 Tokenizer:");
    println!("  Vocabulary:  {:?} tokens", tokenizer.vocab_size());
    println!("  BOS token: {}", tokenizer.bos_token_id);
    println!("  EOS token: {}", tokenizer.eos_token_id);
    println!("  Unknown token: {}", tokenizer.unknown_token_id);
    println!("  Add BOS: {}", tokenizer.add_bos_token);
    println!("  Add EOS: {}", tokenizer.add_eos_token);

    // Sample vocabulary
    if !tokenizer.vocab.is_empty() {
        println!("\n📖 Sample Vocabulary:");
        for i in 0..10.min(tokenizer.vocab.len()) {
            let token = &tokenizer.vocab[i];
            let display = if token.len() > 20 {
                format!("{}...", &token[..17])
            } else {
                token.clone()
            };
            println!("  {}: '{}'", i, display);
        }
    }

    // Tensor summary
    println!("\n🎯 Model Tensors:");
    println!("  Total tensors: {}", loader.tensor_count());

    // File info
    let file_size = std::fs::metadata(model_path)?.len();
    println!("\n💾 File Information:");
    println!("  File size: {:.2} GB", file_size as f64 / 1_000_000_000.0);
    println!("  Load time: {:?}", load_time);
    println!(
        "  Load speed: {:.1} GB/s",
        file_size as f64 / 1_000_000_000.0 / load_time.as_secs_f64()
    );

    Ok(())
}

fn run_generate(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 2 {
        println!("Error: Missing model file or prompt");
        println!("Usage: gguf-cli generate <model.gguf> <prompt> [max_tokens]");
        return Ok(());
    }

    let model_path = &args[0];
    let prompt = &args[1];
    let max_tokens = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);

    println!("🚀 Loading model: {}", model_path);
    let load_start = Instant::now();
    let loader = ModelLoader::load(model_path)?;
    println!("⏱️  Model loaded in: {:?}", load_start.elapsed());

    println!("🧠 Creating inference engine...");
    let engine_start = Instant::now();
    let engine = InferenceEngine::new(loader);
    println!("⏱️  Engine ready in: {:?}", engine_start.elapsed());

    println!("💭 Generating text...");
    println!("📝 Prompt: \"{}\"", prompt);
    println!("🎯 Max tokens: {}", max_tokens);
    println!("{}", "=".repeat(50));

    let gen_start = Instant::now();
    match engine.generate(prompt, max_tokens) {
        Ok(result) => {
            let gen_time = gen_start.elapsed();
            println!("\n📄 Generated Text:");
            println!("{}", result);
            println!("{}", "=".repeat(50));
            println!("⏱️  Generation time: {:?}", gen_time);
            println!(
                "⚡ Average speed: {:.2} tokens/second",
                max_tokens as f64 / gen_time.as_secs_f64()
            );
        }
        Err(e) => {
            println!("❌ Generation failed: {}", e);
            println!("💡 This might be due to:");
            println!("   - Missing tensor names in the model");
            println!("   - Unsupported quantization format");
            println!("   - Model architecture differences");
        }
    }

    Ok(())
}

fn run_chat(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        println!("Error: Missing model file");
        println!("Usage: gguf-cli chat <model.gguf>");
        return Ok(());
    }

    let model_path = &args[0];

    println!("🚀 Loading model: {}", model_path);
    let loader = ModelLoader::load(model_path)?;

    println!("🧠 Creating inference engine...");
    let engine = InferenceEngine::new(loader);

    println!("💬 Interactive Chat Mode");
    println!("Type 'quit' to exit, 'help' for commands");
    println!("{}", "=".repeat(50));

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        match input {
            "quit" | "exit" => {
                println!("Goodbye! 👋");
                break;
            }
            "help" => {
                println!("Commands:");
                println!("  quit/exit - Exit chat mode");
                println!("  help      - Show this help");
                println!("  clear     - Clear screen");
                continue;
            }
            "clear" => {
                print!("\x1B[2J\x1B[1;1H"); // Clear screen
                continue;
            }
            _ => {
                print!("Assistant: ");
                io::stdout().flush()?;

                match engine.generate(input, 50) {
                    Ok(response) => {
                        println!("{}\n", response);
                    }
                    Err(e) => {
                        println!("❌ Error: {}\n", e);
                    }
                }
            }
        }
    }

    Ok(())
}

fn run_benchmark(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        println!("Error: Missing model file");
        println!("Usage: gguf-cli benchmark <model.gguf>");
        return Ok(());
    }

    let model_path = &args[0];

    println!("🏁 Benchmarking GGUF Loader");
    println!("Model: {}", model_path);
    println!("{}", "=".repeat(50));

    // Cold load test
    println!("❄️  Cold Load Test:");
    let cold_start = Instant::now();
    let loader = ModelLoader::load(model_path)?;
    let cold_time = cold_start.elapsed();
    println!("⏱️  Cold load: {:?}", cold_time);

    // Model info
    let config = loader.config();
    println!("\n📊 Model Stats:");
    println!("  Architecture: {}", config.architecture);
    println!("  Size: {}", config.size_label);
    println!("  Tensors: {}", loader.tensor_count());
    println!("  Vocab: {:?} tokens", config.vocab_size);

    drop(loader);

    // Warm load tests
    println!("\n🔥 Warm Load Tests (5 iterations):");
    let mut times = Vec::new();

    for i in 1..=5 {
        let start = Instant::now();
        let loader = ModelLoader::load(model_path)?;
        let time = start.elapsed();
        times.push(time);
        println!("  Load {}: {:?}", i, time);
        drop(loader);
    }

    // Statistics
    let avg = times.iter().sum::<std::time::Duration>() / times.len() as u32;
    let min = times.iter().min().unwrap();
    let max = times.iter().max().unwrap();

    println!("\n📈 Statistics:");
    println!("  Cold load:    {:?}", cold_time);
    println!("  Warm average: {:?}", avg);
    println!("  Warm min:     {:?}", min);
    println!("  Warm max:     {:?}", max);

    // File size analysis
    let file_size = std::fs::metadata(model_path)?.len();
    let gb_size = file_size as f64 / 1_000_000_000.0;
    println!("\n💾 Performance Analysis:");
    println!("  File size:    {:.2} GB", gb_size);
    println!("  Load speed:   {:.1} GB/s", gb_size / avg.as_secs_f64());
    println!("  Memory map:   Zero-copy ✅");

    // Tensor access benchmark
    println!("\n⚡ Tensor Access Benchmark:");
    let loader = ModelLoader::load(model_path)?;

    let lookup_start = Instant::now();
    for _ in 0..1000 {
        let _ = loader.get_tensor("token_emb.weight");
    }
    let lookup_time = lookup_start.elapsed();
    println!("  1000 lookups: {:?}", lookup_time);
    println!("  Per lookup:   {:?}", lookup_time / 1000);

    Ok(())
}

fn run_tensors(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        println!("Error: Missing model file");
        println!("Usage: gguf-cli tensors <model.gguf> [--all]");
        return Ok(());
    }

    let model_path = &args[0];
    let show_all = args.get(1).map_or(false, |arg| arg == "--all");

    println!("🎯 Loading model tensors: {}", model_path);
    let loader = ModelLoader::load(model_path)?;

    println!("📊 Total tensors: {}", loader.tensor_count());

    // Group tensors by type
    let mut tensor_info: Vec<_> = loader
        .tensor_names()
        .map(|name| {
            let tensor = loader.get_tensor(name).unwrap();
            (
                name.clone(),
                tensor.shape().to_vec(),
                tensor.quant_type(),
                tensor.data.len(),
            )
        })
        .collect();

    // Sort by name for consistent output
    tensor_info.sort_by(|a, b| a.0.cmp(&b.0));

    let display_count = if show_all {
        tensor_info.len()
    } else {
        20.min(tensor_info.len())
    };

    println!("{}", "=".repeat(80));
    println!(
        "{:<40} {:<20} {:<10} {:<10}",
        "Tensor Name", "Shape", "Type", "Size (KB)"
    );
    println!("{}", "-".repeat(80));

    // Look for embedding tensors specifically
    let mut embedding_tensors = Vec::new();

    for (i, (name, shape, quant_type, size_bytes)) in tensor_info.iter().enumerate() {
        if i < display_count {
            let shape_str = format!("{:?}", shape);
            let size_kb = size_bytes / 1024;

            // Safe string truncation
            let display_shape = if shape_str.len() > 20 {
                format!("{}...", &shape_str[..17])
            } else {
                shape_str
            };

            println!(
                "{:<40} {:<20} {:<10?} {:<10}",
                name, display_shape, quant_type, size_kb
            );
        }

        // Collect potential embedding tensors
        if name.contains("embed") || name.contains("tok") {
            embedding_tensors.push(name.clone());
        }
    }

    if !show_all && tensor_info.len() > display_count {
        println!(
            "... and {} more tensors (use --all to see all)",
            tensor_info.len() - display_count
        );
    }

    // Show potential embedding tensors
    if !embedding_tensors.is_empty() {
        println!("\n🔍 Potential Token Embedding Tensors:");
        for name in embedding_tensors {
            println!("  {}", name);
        }
    }

    // Summary by tensor type
    println!("\n📈 Tensor Summary:");
    let mut type_counts = std::collections::HashMap::new();
    let mut type_sizes = std::collections::HashMap::new();

    for name in loader.tensor_names() {
        if let Some(tensor) = loader.get_tensor(name) {
            let quant_type = tensor.quant_type();
            *type_counts.entry(quant_type).or_insert(0) += 1;
            *type_sizes.entry(quant_type).or_insert(0u64) += tensor.data.len() as u64;
        }
    }

    for (quant_type, count) in type_counts {
        let size_mb = type_sizes[&quant_type] as f64 / 1_000_000.0;
        println!("  {:?}: {} tensors, {:.1} MB", quant_type, count, size_mb);
    }

    Ok(())
}
