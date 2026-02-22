//! `crabinfer auth` subcommands for managing API keys.

use crabinfer_core::credentials;

const VALID_PROVIDERS: &[&str] = &["openai", "anthropic", "google", "ollama", "vllm"];

fn validate_provider(provider: &str) {
    if !VALID_PROVIDERS.contains(&provider) {
        eprintln!(
            "Error: unknown provider '{}'. Valid providers: {}",
            provider,
            VALID_PROVIDERS.join(", ")
        );
        std::process::exit(1);
    }
}

/// Store an API key for a provider.
pub fn set(provider: &str) {
    validate_provider(provider);

    if provider == "ollama" {
        println!("Ollama does not require an API key.");
        return;
    }

    let key = match rpassword::prompt_password(format!("Enter API key for {}: ", provider)) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Error reading input: {e}");
            std::process::exit(1);
        }
    };
    let key = key.trim();

    if key.is_empty() {
        eprintln!("Error: empty key.");
        std::process::exit(1);
    }

    credentials::set_api_key(provider, key);

    if let Err(e) = save_to_config(provider, key) {
        eprintln!("Warning: stored in memory but failed to save to disk: {e}");
    } else {
        println!("API key for '{}' saved.", provider);
    }
}

/// List configured providers.
pub fn list() {
    load_from_config();
    let providers = credentials::configured_providers();
    if providers.is_empty() {
        println!("No API keys configured.");
        println!("Use 'crabinfer auth set <provider>' to add one.");
        return;
    }
    println!("Configured providers:");
    for p in &providers {
        println!("  - {}", p);
    }
}

/// Remove a stored API key.
pub fn remove(provider: &str) {
    validate_provider(provider);
    load_from_config();

    if !credentials::has_api_key(provider) {
        println!("No API key stored for '{}'.", provider);
        return;
    }

    credentials::remove_api_key(provider);
    if let Err(e) = remove_from_config(provider) {
        eprintln!("Warning: removed from memory but failed to update disk: {e}");
    } else {
        println!("API key for '{}' removed.", provider);
    }
}

/// Validate a stored API key.
pub fn test(provider: &str) {
    validate_provider(provider);
    load_from_config();

    if !credentials::has_api_key(provider) {
        eprintln!(
            "No API key stored for '{}'. Use 'crabinfer auth set {}' first.",
            provider, provider
        );
        std::process::exit(1);
    }

    eprint!("Validating {} key... ", provider);
    match credentials::validate_api_key(provider) {
        Ok(true) => println!("valid!"),
        Ok(false) => {
            println!("INVALID (authentication failed)");
            std::process::exit(1);
        }
        Err(e) => {
            println!("error: {e}");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Config file persistence (~/.config/crabinfer/credentials.toml)
// ---------------------------------------------------------------------------

fn config_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home)
        .join(".config")
        .join("crabinfer")
        .join("credentials.toml")
}

/// Load credentials from config file into the in-memory store.
pub fn load_from_config() {
    let path = config_path();
    if !path.exists() {
        return;
    }
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut in_section = false;
    for line in content.lines() {
        let line = line.trim();
        if line == "[credentials]" {
            in_section = true;
            continue;
        }
        if line.starts_with('[') {
            in_section = false;
            continue;
        }
        if in_section {
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim();
                let val = val.trim().trim_matches('"');
                if !val.is_empty() {
                    credentials::set_api_key(key, val);
                }
            }
        }
    }
}

fn save_to_config(provider: &str, key: &str) -> std::io::Result<()> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Load existing entries, replace/add, write back
    let mut entries: Vec<(String, String)> = Vec::new();
    if path.exists() {
        let content = std::fs::read_to_string(&path)?;
        let mut in_section = false;
        for line in content.lines() {
            let lt = line.trim();
            if lt == "[credentials]" {
                in_section = true;
                continue;
            }
            if lt.starts_with('[') {
                in_section = false;
                continue;
            }
            if in_section {
                if let Some((k, v)) = lt.split_once('=') {
                    let k = k.trim().to_string();
                    let v = v.trim().trim_matches('"').to_string();
                    if k != provider && !v.is_empty() {
                        entries.push((k, v));
                    }
                }
            }
        }
    }
    entries.push((provider.to_string(), key.to_string()));

    let mut out = String::from("[credentials]\n");
    for (k, v) in &entries {
        out.push_str(&format!("{} = \"{}\"\n", k, v));
    }
    std::fs::write(&path, &out)?;

    // Set file permissions to owner-only (0600) on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }

    Ok(())
}

fn remove_from_config(provider: &str) -> std::io::Result<()> {
    let path = config_path();
    if !path.exists() {
        return Ok(());
    }

    // Load existing, filter out the provider, write back
    let mut entries: Vec<(String, String)> = Vec::new();
    let content = std::fs::read_to_string(&path)?;
    let mut in_section = false;
    for line in content.lines() {
        let lt = line.trim();
        if lt == "[credentials]" {
            in_section = true;
            continue;
        }
        if lt.starts_with('[') {
            in_section = false;
            continue;
        }
        if in_section {
            if let Some((k, v)) = lt.split_once('=') {
                let k = k.trim().to_string();
                let v = v.trim().trim_matches('"').to_string();
                if k != provider && !v.is_empty() {
                    entries.push((k, v));
                }
            }
        }
    }

    if entries.is_empty() {
        // No credentials left — remove the file
        std::fs::remove_file(&path)?;
    } else {
        let mut out = String::from("[credentials]\n");
        for (k, v) in &entries {
            out.push_str(&format!("{} = \"{}\"\n", k, v));
        }
        std::fs::write(&path, &out)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
    }

    Ok(())
}
