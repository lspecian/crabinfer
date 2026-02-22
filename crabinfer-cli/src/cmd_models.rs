use crabinfer_core::catalog::{self, ModelCategory};
use crabinfer_core::download::ModelDownloadManager;

/// Format bytes as human-readable size.
fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

/// List models in the catalog.
pub fn list(compatible: bool, category: Option<&str>) {
    let mut models = if compatible {
        let device = crabinfer_core::detect_device();
        catalog::for_device(&device)
    } else {
        catalog::all()
    };

    if let Some(cat_str) = category {
        let cat = match cat_str {
            "code" => ModelCategory::Code,
            "reasoning" => ModelCategory::Reasoning,
            _ => ModelCategory::General,
        };
        models.retain(|m| m.category == cat);
    }

    if models.is_empty() {
        println!("No models found.");
        return;
    }

    println!(
        "{:<35} {:>8} {:>6} {:>8} {:>8}",
        "MODEL", "SIZE", "PARAMS", "QUANT", "CATEGORY"
    );
    println!("{}", "-".repeat(72));

    for m in &models {
        let cat = match m.category {
            ModelCategory::General => "general",
            ModelCategory::Code => "code",
            ModelCategory::Reasoning => "reason",
        };
        let params = if m.is_moe {
            format!("{}B/{}B", m.param_count_b, m.active_param_count_b)
        } else {
            format!("{}B", m.param_count_b)
        };
        println!(
            "{:<35} {:>8} {:>6} {:>8} {:>8}",
            m.id,
            format_size(m.size_bytes),
            params,
            m.quant,
            cat
        );
    }
    println!("\n{} model(s)", models.len());
}

/// Show recommended models for this device.
pub fn recommend() {
    let device = crabinfer_core::detect_device();
    println!(
        "Device: {} ({} RAM, {})",
        device.chip_name,
        format_size(device.total_memory_bytes),
        device.recommended_quant
    );
    println!();

    let recs = catalog::recommended(&device);
    if recs.is_empty() {
        println!("No models fit on this device.");
        return;
    }

    for m in &recs {
        let cat = match m.category {
            ModelCategory::General => "General",
            ModelCategory::Code => "Code",
            ModelCategory::Reasoning => "Reasoning",
        };
        let moe_tag = if m.is_moe { " (MoE)" } else { "" };
        println!(
            "  {} [{}]: {} — {}B{} {} ({})",
            cat,
            m.id,
            m.name,
            m.param_count_b,
            moe_tag,
            m.quant,
            format_size(m.size_bytes)
        );
    }
    println!("\nPull with: crabinfer models pull <id>");
}

/// Download a model.
pub fn pull(id: &str) {
    let entry = match catalog::get(id) {
        Some(e) => e,
        None => {
            eprintln!("Error: model '{}' not found in catalog.", id);
            eprintln!("Run 'crabinfer models list' to see available models.");
            std::process::exit(1);
        }
    };

    if entry.requires_auth {
        eprintln!(
            "Note: {} may require HuggingFace authentication.",
            entry.hf_repo
        );
        eprintln!("Set HF_TOKEN environment variable if download fails.\n");
    }

    let mgr = match ModelDownloadManager::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error creating download manager: {e}");
            std::process::exit(1);
        }
    };

    if mgr.is_downloaded(id.to_string()) {
        println!("Model '{}' is already downloaded.", id);
        if let Some(path) = mgr.model_path(id.to_string()) {
            println!("  Path: {}", path);
        }
        return;
    }

    println!(
        "Downloading {} ({})...",
        entry.name,
        format_size(entry.size_bytes)
    );

    struct CliProgressListener;
    impl crabinfer_core::download::DownloadProgressListener for CliProgressListener {
        fn on_progress(&self, bytes_downloaded: u64, bytes_total: u64, phase: String) {
            if bytes_total > 0 && phase == "downloading_model" {
                let pct = (bytes_downloaded as f64 / bytes_total as f64 * 100.0) as u32;
                let downloaded = format_size(bytes_downloaded);
                let total = format_size(bytes_total);
                eprint!("\r  {downloaded} / {total} ({pct}%)    ");
            } else if phase == "verifying" {
                eprint!("\r  Verifying integrity...            ");
            } else if phase == "downloading_tokenizer" {
                eprint!("\r  Downloading tokenizer...          ");
            } else if phase == "complete" {
                eprintln!("\r  Done!                             ");
            }
        }
    }

    match mgr.download(entry, Some(Box::new(CliProgressListener))) {
        Ok(result) => {
            println!("Model:     {}", result.model_path);
            println!("Tokenizer: {}", result.tokenizer_path);
        }
        Err(e) => {
            eprintln!("\nDownload failed: {e}");
            std::process::exit(1);
        }
    }
}

/// List downloaded models.
pub fn downloaded() {
    let mgr = match ModelDownloadManager::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    let models = mgr.list_downloaded();
    if models.is_empty() {
        println!("No models downloaded.");
        println!("Run 'crabinfer models pull <id>' to download a model.");
        return;
    }

    println!("{:<35} {:>8} {}", "MODEL", "SIZE", "PATH");
    println!("{}", "-".repeat(72));
    for m in &models {
        println!(
            "{:<35} {:>8} {}",
            m.id,
            format_size(m.size_bytes),
            m.model_path
        );
    }
    println!(
        "\n{} model(s), {} total",
        models.len(),
        format_size(mgr.total_disk_usage())
    );
}

/// Remove a downloaded model.
pub fn remove(id: &str) {
    let mgr = match ModelDownloadManager::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    if !mgr.is_downloaded(id.to_string()) {
        println!("Model '{}' is not downloaded.", id);
        return;
    }

    match mgr.delete(id.to_string()) {
        Ok(()) => println!("Removed model '{}'.", id),
        Err(e) => {
            eprintln!("Error removing model: {e}");
            std::process::exit(1);
        }
    }
}

/// Show storage information.
pub fn storage() {
    let mgr = match ModelDownloadManager::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    println!("Storage directory: {}", mgr.storage_directory());
    println!("Total disk usage:  {}", format_size(mgr.total_disk_usage()));
    println!(
        "Downloaded models: {}",
        mgr.list_downloaded().len()
    );
}
