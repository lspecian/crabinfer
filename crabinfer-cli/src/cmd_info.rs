use crabinfer_core::device::detect_device;

pub fn run() {
    let info = detect_device();

    println!("CrabInfer Device Info");
    println!("=====================");
    println!();
    println!("Chip:            {}", info.chip_name);
    if !info.chip_variant.is_empty() {
        println!("Variant:         {}", info.chip_variant);
    }
    println!("Device model:    {}", info.device_model);
    println!("Metal GPU:       {}", if info.has_metal_gpu { "Yes" } else { "No" });
    println!("Neural Engine:   {}", if info.has_neural_engine { "Yes" } else { "No" });
    println!();
    println!("Memory");
    println!("------");
    println!("Total:           {} GB", info.total_memory_bytes / (1024 * 1024 * 1024));
    println!("Available:       {} GB", info.available_memory_bytes / (1024 * 1024 * 1024));
    println!("Max model file:  {} GB", info.max_model_file_size_bytes / (1024 * 1024 * 1024));
    println!();
    println!("Recommendations");
    println!("---------------");
    println!("Quantization:    {}", info.recommended_quant);
    println!("Max model size:  {}B parameters", info.max_model_size_b);
    println!("Context length:  {}", info.recommended_context_length);
}
