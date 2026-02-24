//! `crabinfer assistant` — interactive AI assistant with tool calling.
//!
//! Uses the Agent runtime to provide an autonomous assistant that can
//! read/write files, run shell commands, search knowledge, and remember
//! facts across sessions.

use crabinfer_core::agent::{Agent, AgentConfig};
use crabinfer_core::conversation::ConversationMemory;
use crabinfer_core::facts::MemoryStore;
use crabinfer_core::knowledge::KnowledgeBase;
use crabinfer_core::prompt::SystemPrompt;
use crabinfer_core::provider::Provider;
use crabinfer_core::CrabInferError;
use std::sync::Arc;

/// Run the interactive assistant REPL.
pub fn run(
    provider_type: &str,
    model: &str,
    max_tokens: u32,
    temperature: f32,
    data_dir: Option<&str>,
    knowledge_files: &[String],
) {
    // Set up data directory for persistence
    let data_dir = data_dir
        .map(|d| d.to_string())
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            format!("{}/.crabinfer/assistant", home)
        });
    std::fs::create_dir_all(&data_dir).ok();

    // Create the provider
    let provider: Arc<dyn Provider> = match create_provider(provider_type, model) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to create provider: {e}");
            eprintln!("Hint: run `crabinfer auth set {provider_type}` to configure API key");
            std::process::exit(1);
        }
    };

    // Load or create persistent state
    let conv_path = format!("{}/conversation.json", data_dir);
    let conversation = ConversationMemory::load(&conv_path)
        .unwrap_or_else(|_| ConversationMemory::new("assistant").with_persist_path(&conv_path));

    let facts_path = format!("{}/facts.json", data_dir);
    let facts = MemoryStore::load(&facts_path)
        .unwrap_or_else(|_| MemoryStore::new())
        .with_persist_path(&facts_path);

    // Build system prompt
    let system_prompt = SystemPrompt::new()
        .identity("You are CrabInfer Assistant, a helpful AI that can read files, write files, run commands, and fetch web pages.")
        .instruction("When the user asks you to do something, use the available tools to help them.")
        .instruction("Think step by step. Use tools when needed, then provide a clear summary.")
        .instruction("If you're unsure about something, ask the user for clarification instead of guessing.");

    // Configure agent
    let config = AgentConfig {
        max_tool_rounds: 10,
        max_tokens,
        temperature,
        top_p: 0.9,
        rag_top_k: 3,
    };

    let mut agent = Agent::new(provider)
        .with_system_prompt(system_prompt)
        .with_conversation(conversation)
        .with_facts(facts)
        .with_config(config);

    // Load knowledge files if any
    if !knowledge_files.is_empty() {
        let embedder = Box::new(crabinfer_core::embedding::TfIdfEmbedder::default());
        let mut kb = KnowledgeBase::new(embedder);
        for path in knowledge_files {
            match kb.add_file(path) {
                Ok(chunks) => eprintln!("  Indexed {path} ({chunks} chunks)"),
                Err(e) => eprintln!("  Failed to index {path}: {e}"),
            }
        }
        if kb.chunk_count() > 0 {
            agent = agent.with_knowledge(kb);
        }
    }

    // Connect to configured MCP servers
    let mcp_registry = crabinfer_core::mcp::McpServerRegistry::load_default();
    let mcp_connected = mcp_registry.connect_all(agent.tools_mut());

    // Print banner
    eprintln!("CrabInfer Assistant");
    eprintln!("  Provider: {provider_type}");
    eprintln!("  Model: {model}");
    eprintln!("  Tools: {}", agent.tools_mut().tool_names().join(", "));
    if !mcp_connected.is_empty() {
        for (name, count) in &mcp_connected {
            eprintln!("  MCP: {name} ({count} tools)");
        }
    }
    if !knowledge_files.is_empty() {
        eprintln!("  Knowledge: {} files indexed", knowledge_files.len());
    }
    eprintln!();
    eprintln!("Commands: /help, /tools, /facts, /clear, /save, /quit");
    eprintln!();

    // Start REPL
    let mut rl = match rustyline::DefaultEditor::new() {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Failed to initialize line editor: {e}");
            std::process::exit(1);
        }
    };

    loop {
        let input = match rl.readline("you> ") {
            Ok(line) => line,
            Err(
                rustyline::error::ReadlineError::Interrupted
                | rustyline::error::ReadlineError::Eof,
            ) => break,
            Err(e) => {
                eprintln!("Input error: {e}");
                break;
            }
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Handle commands
        match trimmed {
            "/quit" | "/exit" => break,
            "/clear" => {
                agent.clear_conversation();
                eprintln!("[Conversation cleared]");
                continue;
            }
            "/save" => {
                match agent.save() {
                    Ok(()) => eprintln!("[State saved]"),
                    Err(e) => eprintln!("[Save failed: {e}]"),
                }
                continue;
            }
            "/tools" => {
                eprintln!("Available tools:");
                for name in agent.tools_mut().tool_names() {
                    eprintln!("  - {name}");
                }
                continue;
            }
            "/facts" => {
                let facts = agent.facts_mut().as_prompt_context();
                if facts.is_empty() {
                    eprintln!("[No facts stored]");
                } else {
                    eprintln!("Stored facts:");
                    for fact in &facts {
                        eprintln!("  {fact}");
                    }
                }
                continue;
            }
            "/help" => {
                eprintln!("Commands:");
                eprintln!("  /tools  — List available tools");
                eprintln!("  /facts  — Show stored facts");
                eprintln!("  /clear  — Clear conversation history");
                eprintln!("  /save   — Save state to disk");
                eprintln!("  /quit   — Exit assistant");
                eprintln!("  /help   — Show this help");
                eprintln!();
                eprintln!("Special commands:");
                eprintln!("  /remember <key> = <value>  — Store a fact");
                eprintln!("  /forget <key>              — Remove a fact");
                continue;
            }
            _ => {}
        }

        // Handle /remember and /forget
        if trimmed.starts_with("/remember ") {
            let rest = &trimmed["/remember ".len()..];
            if let Some((key, value)) = rest.split_once('=') {
                agent.facts_mut().add_fact(key.trim(), value.trim());
                eprintln!("[Remembered: {} = {}]", key.trim(), value.trim());
            } else {
                eprintln!("Usage: /remember <key> = <value>");
            }
            continue;
        }
        if trimmed.starts_with("/forget ") {
            let key = trimmed["/forget ".len()..].trim();
            if agent.facts_mut().remove_fact(key) {
                eprintln!("[Forgot: {key}]");
            } else {
                eprintln!("[No fact with key '{key}']");
            }
            continue;
        }

        let _ = rl.add_history_entry(trimmed);

        // Run the agent
        match agent.run(trimmed) {
            Ok(response) => {
                // Print tool call activity
                if !response.tool_calls.is_empty() {
                    for tc in &response.tool_calls {
                        let status = if tc.is_error { "ERROR" } else { "OK" };
                        eprintln!("  [{status}] {}", tc.tool_name);
                    }
                    eprintln!();
                }

                // Print the response
                println!("{}", response.text);

                // Print stats
                eprintln!(
                    "[{} rounds, {} tool calls]",
                    response.rounds,
                    response.tool_calls.len()
                );
                eprintln!();
            }
            Err(e) => {
                eprintln!("Error: {e}");
                eprintln!();
            }
        }
    }

    // Auto-save on exit
    eprintln!();
    match agent.save() {
        Ok(()) => eprintln!("State saved. Goodbye!"),
        Err(_) => eprintln!("Goodbye!"),
    }
}

/// Create a provider from the given type and model.
fn create_provider(
    provider_type: &str,
    model: &str,
) -> Result<Arc<dyn Provider>, CrabInferError> {
    let api_key =
        crabinfer_core::credentials::get_api_key(provider_type).unwrap_or_default();

    if api_key.is_empty() && !matches!(provider_type, "ollama" | "local") {
        return Err(CrabInferError::AuthenticationFailed {
            provider: provider_type.to_string(),
        });
    }

    let config = crabinfer_core::provider::ProviderConfig {
        provider_type: provider_type.to_string(),
        api_key,
        base_url: String::new(),
        default_model: model.to_string(),
        timeout_seconds: 60,
        tier_override: String::new(),
    };

    let provider: Box<dyn Provider> = match provider_type {
        "openai" => Box::new(crabinfer_core::providers::openai::OpenAIProvider::new(
            config,
        )?),
        "anthropic" => Box::new(
            crabinfer_core::providers::anthropic::AnthropicProvider::new(config)?,
        ),
        "google" => Box::new(crabinfer_core::providers::google::GoogleProvider::new(
            config,
        )?),
        "ollama" => Box::new(crabinfer_core::providers::ollama::OllamaProvider::new(
            config,
        )?),
        "vllm" => Box::new(crabinfer_core::providers::vllm::VllmProvider::new(config)?),
        _ => {
            return Err(CrabInferError::ProviderNotAvailable {
                provider: provider_type.to_string(),
            })
        }
    };

    Ok(Arc::from(provider))
}
