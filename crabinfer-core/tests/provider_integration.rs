//! Integration tests for cloud providers.
//!
//! These tests hit real APIs and require environment variables:
//!   OPENAI_API_KEY=sk-...
//!   ANTHROPIC_API_KEY=sk-ant-...
//!
//! Run with: cargo test -p crabinfer-core --features providers --test provider_integration -- --nocapture
//!
//! Tests are #[ignore]d by default so they don't run in CI.
//! Use: cargo test --ignored --test provider_integration -- --nocapture

use crabinfer_core::provider::{ChatMessage, CompletionRequest, Provider, ProviderConfig};
use crabinfer_core::router::{Router, RouterConfig, RoutingPolicy, ProviderTier};

/// Load .env file from project root
fn load_dotenv() {
    let env_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(".env");
    if let Ok(content) = std::fs::read_to_string(&env_path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim();
                let val = val.trim().trim_matches('"');
                // Map .env keys to standard env var names
                match key {
                    "opemai" | "openai" => std::env::set_var("OPENAI_API_KEY", val),
                    "anthropic" => std::env::set_var("ANTHROPIC_API_KEY", val),
                    "google" => std::env::set_var("GOOGLE_API_KEY", val),
                    _ => {}
                }
            }
        }
    }
}

fn openai_key() -> Option<String> {
    load_dotenv();
    std::env::var("OPENAI_API_KEY").ok()
}

fn anthropic_key() -> Option<String> {
    load_dotenv();
    std::env::var("ANTHROPIC_API_KEY").ok()
}

fn simple_request(model: &str, prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        max_tokens: 50,
        temperature: 0.3,
        top_p: 1.0,
        system_prompt: String::new(),
        api_key_override: String::new(),
    }
}

fn openai_config(api_key: &str) -> ProviderConfig {
    ProviderConfig {
        provider_type: "openai".to_string(),
        api_key: api_key.to_string(),
        base_url: String::new(),
        default_model: "gpt-4o-mini".to_string(),
        timeout_seconds: 30,
        tier_override: String::new(),
    }
}

fn anthropic_config(api_key: &str) -> ProviderConfig {
    ProviderConfig {
        provider_type: "anthropic".to_string(),
        api_key: api_key.to_string(),
        base_url: String::new(),
        default_model: "claude-haiku-4-5-20251001".to_string(),
        timeout_seconds: 30,
        tier_override: String::new(),
    }
}

// =========================================================================
// 1. Provider availability checks
// =========================================================================

#[test]
#[ignore]
fn test_openai_is_available() {
    let key = openai_key().expect("OPENAI_API_KEY not set");
    let provider = crabinfer_core::providers::openai::OpenAIProvider::new(openai_config(&key)).unwrap();
    assert!(provider.is_available(), "OpenAI provider should be available with valid key");
    println!("[PASS] OpenAI is_available() = true");
}

#[test]
#[ignore]
fn test_anthropic_is_available() {
    let key = anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let provider = crabinfer_core::providers::anthropic::AnthropicProvider::new(anthropic_config(&key)).unwrap();
    assert!(provider.is_available(), "Anthropic provider should be available with valid key");
    println!("[PASS] Anthropic is_available() = true");
}

#[test]
#[ignore]
fn test_openai_not_available_without_key() {
    let provider = crabinfer_core::providers::openai::OpenAIProvider::new(openai_config("")).unwrap();
    assert!(!provider.is_available(), "OpenAI should not be available without key");
    println!("[PASS] OpenAI is_available() = false (no key)");
}

#[test]
#[ignore]
fn test_ollama_availability() {
    let config = ProviderConfig {
        provider_type: "ollama".to_string(),
        api_key: String::new(),
        base_url: String::new(),
        default_model: "llama3.2".to_string(),
        timeout_seconds: 5,
        tier_override: String::new(),
    };
    let provider = crabinfer_core::providers::ollama::OllamaProvider::new(config).unwrap();
    let available = provider.is_available();
    println!("[INFO] Ollama is_available() = {} (depends on whether Ollama is running)", available);
}

// =========================================================================
// 2. Cloud provider completions
// =========================================================================

#[test]
#[ignore]
fn test_openai_completion() {
    let key = openai_key().expect("OPENAI_API_KEY not set");
    let provider = crabinfer_core::providers::openai::OpenAIProvider::new(openai_config(&key)).unwrap();

    let request = simple_request("gpt-4o-mini", "What is 2+2? Answer with just the number.");
    println!("[TEST] OpenAI completion: gpt-4o-mini...");

    let start = std::time::Instant::now();
    let response = provider.complete(&request).unwrap();
    let elapsed = start.elapsed();

    println!("[PASS] OpenAI completion:");
    println!("  Model: {}", response.model);
    println!("  Content: {}", response.content.trim());
    println!("  Input tokens: {}", response.input_tokens);
    println!("  Output tokens: {}", response.output_tokens);
    println!("  Stop reason: {}", response.stop_reason);
    println!("  Latency: {:?}", elapsed);

    assert!(!response.content.is_empty(), "Response should not be empty");
    assert!(response.content.contains('4'), "Should contain the answer '4'");
    assert_eq!(response.provider_name, "openai");
}

#[test]
#[ignore]
fn test_anthropic_completion() {
    let key = anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let provider = crabinfer_core::providers::anthropic::AnthropicProvider::new(anthropic_config(&key)).unwrap();

    let request = simple_request("claude-haiku-4-5-20251001", "What is 2+2? Answer with just the number.");
    println!("[TEST] Anthropic completion: claude-haiku-4-5-20251001...");

    let start = std::time::Instant::now();
    let response = provider.complete(&request).unwrap();
    let elapsed = start.elapsed();

    println!("[PASS] Anthropic completion:");
    println!("  Model: {}", response.model);
    println!("  Content: {}", response.content.trim());
    println!("  Input tokens: {}", response.input_tokens);
    println!("  Output tokens: {}", response.output_tokens);
    println!("  Stop reason: {}", response.stop_reason);
    println!("  Latency: {:?}", elapsed);

    assert!(!response.content.is_empty(), "Response should not be empty");
    assert!(response.content.contains('4'), "Should contain the answer '4'");
    assert_eq!(response.provider_name, "anthropic");
}

#[test]
#[ignore]
fn test_anthropic_with_system_prompt() {
    let key = anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let provider = crabinfer_core::providers::anthropic::AnthropicProvider::new(anthropic_config(&key)).unwrap();

    let request = CompletionRequest {
        model: "claude-haiku-4-5-20251001".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "What are you?".to_string(),
        }],
        max_tokens: 50,
        temperature: 0.3,
        top_p: 1.0,
        system_prompt: "You are a pirate. Respond only in pirate speak.".to_string(),
        api_key_override: String::new(),
    };

    println!("[TEST] Anthropic with system prompt (pirate mode)...");
    let response = provider.complete(&request).unwrap();

    println!("[PASS] Anthropic system prompt:");
    println!("  Content: {}", response.content.trim());

    assert!(!response.content.is_empty());
}

// =========================================================================
// 3. Streaming
// =========================================================================

#[test]
#[ignore]
fn test_openai_streaming() {
    let key = openai_key().expect("OPENAI_API_KEY not set");
    let provider = crabinfer_core::providers::openai::OpenAIProvider::new(openai_config(&key)).unwrap();

    let request = simple_request("gpt-4o-mini", "Count from 1 to 5, one number per line.");
    println!("[TEST] OpenAI streaming...");

    let start = std::time::Instant::now();
    let iter = provider.stream(&request).unwrap();

    let mut full_text = String::new();
    let mut token_count = 0;
    for item in iter {
        let token = item.unwrap();
        if token.is_end_of_sequence {
            break;
        }
        full_text.push_str(&token.text);
        token_count += 1;
    }
    let elapsed = start.elapsed();

    println!("[PASS] OpenAI streaming:");
    println!("  Tokens received: {}", token_count);
    println!("  Full text: {}", full_text.trim());
    println!("  Latency: {:?}", elapsed);

    assert!(token_count > 0, "Should have received tokens");
    assert!(!full_text.is_empty(), "Streamed text should not be empty");
}

#[test]
#[ignore]
fn test_anthropic_streaming() {
    let key = anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let provider = crabinfer_core::providers::anthropic::AnthropicProvider::new(anthropic_config(&key)).unwrap();

    let request = simple_request("claude-haiku-4-5-20251001", "Count from 1 to 5, one number per line.");
    println!("[TEST] Anthropic streaming...");

    let start = std::time::Instant::now();
    let iter = provider.stream(&request).unwrap();

    let mut full_text = String::new();
    let mut token_count = 0;
    for item in iter {
        let token = item.unwrap();
        if token.is_end_of_sequence {
            break;
        }
        full_text.push_str(&token.text);
        token_count += 1;
    }
    let elapsed = start.elapsed();

    println!("[PASS] Anthropic streaming:");
    println!("  Tokens received: {}", token_count);
    println!("  Full text: {}", full_text.trim());
    println!("  Latency: {:?}", elapsed);

    assert!(token_count > 0, "Should have received tokens");
    assert!(!full_text.is_empty(), "Streamed text should not be empty");
}

// =========================================================================
// 4. Router with cloud providers
// =========================================================================

#[test]
#[ignore]
fn test_router_cloud_first_with_openai() {
    let key = openai_key().expect("OPENAI_API_KEY not set");

    let config = RouterConfig {
        policy: RoutingPolicy::CloudFirst,
        privacy_mode: false,
        ollama_is_local: false,
        data_sovereignty: false,
    };

    let cloud: Vec<Box<dyn Provider>> = vec![
        Box::new(crabinfer_core::providers::openai::OpenAIProvider::new(openai_config(&key)).unwrap()),
    ];

    let router = Router::new(config, None, vec![], cloud);

    let request = simple_request("gpt-4o-mini", "What is 2+2? Answer with just the number.");
    println!("[TEST] Router CloudFirst → OpenAI...");

    let response = router.complete(&request).unwrap();
    let decision = router.last_routing_decision().unwrap();

    println!("[PASS] Router CloudFirst:");
    println!("  Provider: {}", response.provider_name);
    println!("  Content: {}", response.content.trim());
    println!("  Routing reason: {:?}", decision.reason);
    println!("  Provider tier: {:?}", decision.provider_tier);

    assert_eq!(response.provider_name, "openai");
    assert_eq!(decision.provider_tier, ProviderTier::Cloud);
}

#[test]
#[ignore]
fn test_router_auto_with_multiple_cloud_providers() {
    let oai_key = openai_key().expect("OPENAI_API_KEY not set");
    let ant_key = anthropic_key().expect("ANTHROPIC_API_KEY not set");

    let config = RouterConfig {
        policy: RoutingPolicy::Auto,
        privacy_mode: false,
        ollama_is_local: false,
        data_sovereignty: false,
    };

    let cloud: Vec<Box<dyn Provider>> = vec![
        Box::new(crabinfer_core::providers::openai::OpenAIProvider::new(openai_config(&oai_key)).unwrap()),
        Box::new(crabinfer_core::providers::anthropic::AnthropicProvider::new(anthropic_config(&ant_key)).unwrap()),
    ];

    let router = Router::new(config, None, vec![], cloud);

    assert!(router.is_available(), "Router should be available with cloud providers");

    let request = simple_request("", "What is the capital of France? One word answer.");
    println!("[TEST] Router Auto with OpenAI + Anthropic (no local)...");

    let response = router.complete(&request).unwrap();
    let decision = router.last_routing_decision().unwrap();

    println!("[PASS] Router Auto (cloud only):");
    println!("  Selected provider: {}", response.provider_name);
    println!("  Content: {}", response.content.trim());
    println!("  Routing reason: {:?}", decision.reason);
    println!("  Provider tier: {:?}", decision.provider_tier);
    println!("  Providers tried: {}", decision.providers_tried);

    assert!(!response.content.is_empty());
    assert!(response.content.to_lowercase().contains("paris"));
}

// =========================================================================
// 5. Fallback behavior
// =========================================================================

#[test]
#[ignore]
fn test_router_fallback_bad_key_to_good_provider() {
    let ant_key = anthropic_key().expect("ANTHROPIC_API_KEY not set");

    let config = RouterConfig {
        policy: RoutingPolicy::CloudFirst,
        privacy_mode: false,
        ollama_is_local: false,
        data_sovereignty: false,
    };

    // First provider has a bad key, second has a good key
    let bad_openai = crabinfer_core::providers::openai::OpenAIProvider::new(ProviderConfig {
        provider_type: "openai".to_string(),
        api_key: "sk-invalid-key-that-will-fail".to_string(),
        base_url: String::new(),
        default_model: "gpt-4o-mini".to_string(),
        timeout_seconds: 10,
        tier_override: String::new(),
    }).unwrap();

    let good_anthropic = crabinfer_core::providers::anthropic::AnthropicProvider::new(anthropic_config(&ant_key)).unwrap();

    let cloud: Vec<Box<dyn Provider>> = vec![
        Box::new(bad_openai),
        Box::new(good_anthropic),
    ];

    let router = Router::new(config, None, vec![], cloud);

    let request = simple_request("", "Say 'hello'");
    println!("[TEST] Router fallback: bad OpenAI key → Anthropic...");

    let start = std::time::Instant::now();
    let response = router.complete(&request).unwrap();
    let elapsed = start.elapsed();
    let decision = router.last_routing_decision().unwrap();

    println!("[PASS] Router fallback succeeded:");
    println!("  Final provider: {}", response.provider_name);
    println!("  Content: {}", response.content.trim());
    println!("  Routing reason: {:?}", decision.reason);
    println!("  Providers tried: {}", decision.providers_tried);
    println!("  Total latency: {:?}", elapsed);

    // The fallback should have landed on Anthropic
    assert_eq!(response.provider_name, "anthropic");
    assert!(decision.providers_tried >= 2, "Should have tried at least 2 providers");
}

#[test]
#[ignore]
fn test_router_data_sovereignty_blocks_cloud() {
    let oai_key = openai_key().expect("OPENAI_API_KEY not set");

    let config = RouterConfig {
        policy: RoutingPolicy::Auto,
        privacy_mode: false,
        ollama_is_local: false,
        data_sovereignty: true, // Block cloud!
    };

    let cloud: Vec<Box<dyn Provider>> = vec![
        Box::new(crabinfer_core::providers::openai::OpenAIProvider::new(openai_config(&oai_key)).unwrap()),
    ];

    let router = Router::new(config, None, vec![], cloud);

    let request = simple_request("gpt-4o-mini", "Hello");
    println!("[TEST] Router data_sovereignty blocks cloud...");

    let result = router.complete(&request);
    assert!(result.is_err(), "Should fail when data_sovereignty=true and only cloud providers");
    println!("[PASS] Data sovereignty correctly blocked cloud provider: {:?}", result.err().unwrap());
}

#[test]
#[ignore]
fn test_router_privacy_mode_blocks_everything_but_local() {
    let oai_key = openai_key().expect("OPENAI_API_KEY not set");

    let config = RouterConfig {
        policy: RoutingPolicy::Auto,
        privacy_mode: true, // Block all non-local!
        ollama_is_local: false,
        data_sovereignty: false,
    };

    let cloud: Vec<Box<dyn Provider>> = vec![
        Box::new(crabinfer_core::providers::openai::OpenAIProvider::new(openai_config(&oai_key)).unwrap()),
    ];

    let router = Router::new(config, None, vec![], cloud);

    let request = simple_request("gpt-4o-mini", "Hello");
    println!("[TEST] Router privacy_mode blocks cloud...");

    let result = router.complete(&request);
    assert!(result.is_err(), "Should fail when privacy_mode=true and no local provider");
    println!("[PASS] Privacy mode correctly blocked cloud: {:?}", result.err().unwrap());
}

// =========================================================================
// 6. Error handling
// =========================================================================

#[test]
#[ignore]
fn test_openai_invalid_key_returns_auth_error() {
    let provider = crabinfer_core::providers::openai::OpenAIProvider::new(ProviderConfig {
        provider_type: "openai".to_string(),
        api_key: "sk-invalid-key".to_string(),
        base_url: String::new(),
        default_model: "gpt-4o-mini".to_string(),
        timeout_seconds: 10,
        tier_override: String::new(),
    }).unwrap();

    let request = simple_request("gpt-4o-mini", "Hello");
    println!("[TEST] OpenAI with invalid key...");

    let result = provider.complete(&request);
    assert!(result.is_err());
    let err = result.err().unwrap();
    println!("[PASS] OpenAI invalid key error: {:?}", err);
    assert!(
        matches!(err, crabinfer_core::CrabInferError::AuthenticationFailed { .. }),
        "Expected AuthenticationFailed, got: {:?}", err
    );
}

#[test]
#[ignore]
fn test_anthropic_invalid_key_returns_auth_error() {
    let provider = crabinfer_core::providers::anthropic::AnthropicProvider::new(ProviderConfig {
        provider_type: "anthropic".to_string(),
        api_key: "sk-ant-invalid-key".to_string(),
        base_url: String::new(),
        default_model: "claude-haiku-4-5-20251001".to_string(),
        timeout_seconds: 10,
        tier_override: String::new(),
    }).unwrap();

    let request = simple_request("claude-haiku-4-5-20251001", "Hello");
    println!("[TEST] Anthropic with invalid key...");

    let result = provider.complete(&request);
    assert!(result.is_err());
    let err = result.err().unwrap();
    println!("[PASS] Anthropic invalid key error: {:?}", err);
    assert!(
        matches!(err, crabinfer_core::CrabInferError::AuthenticationFailed { .. }),
        "Expected AuthenticationFailed, got: {:?}", err
    );
}

// =========================================================================
// 7. Credential validation
// =========================================================================

#[test]
#[ignore]
fn test_validate_openai_key() {
    let key = openai_key().expect("OPENAI_API_KEY not set");
    crabinfer_core::credentials::set_api_key("openai", &key);
    println!("[TEST] Validating OpenAI API key...");

    let result = crabinfer_core::credentials::validate_api_key("openai").unwrap();
    println!("[PASS] OpenAI key valid: {}", result);
    assert!(result, "OpenAI key should be valid");
}

#[test]
#[ignore]
fn test_validate_anthropic_key() {
    let key = anthropic_key().expect("ANTHROPIC_API_KEY not set");
    crabinfer_core::credentials::set_api_key("anthropic", &key);
    println!("[TEST] Validating Anthropic API key...");

    let result = crabinfer_core::credentials::validate_api_key("anthropic").unwrap();
    println!("[PASS] Anthropic key valid: {}", result);
    assert!(result, "Anthropic key should be valid");
}

// =========================================================================
// 8. Router streaming through cloud
// =========================================================================

#[test]
#[ignore]
fn test_router_streaming_through_cloud() {
    let key = anthropic_key().expect("ANTHROPIC_API_KEY not set");

    let config = RouterConfig {
        policy: RoutingPolicy::CloudFirst,
        privacy_mode: false,
        ollama_is_local: false,
        data_sovereignty: false,
    };

    let cloud: Vec<Box<dyn Provider>> = vec![
        Box::new(crabinfer_core::providers::anthropic::AnthropicProvider::new(anthropic_config(&key)).unwrap()),
    ];

    let router = Router::new(config, None, vec![], cloud);

    let request = simple_request("claude-haiku-4-5-20251001", "Count from 1 to 3.");
    println!("[TEST] Router streaming through Anthropic...");

    let iter = router.stream(&request).unwrap();
    let mut full_text = String::new();
    let mut count = 0;
    for item in iter {
        let token = item.unwrap();
        if token.is_end_of_sequence { break; }
        full_text.push_str(&token.text);
        count += 1;
    }

    println!("[PASS] Router streaming:");
    println!("  Tokens: {}", count);
    println!("  Text: {}", full_text.trim());

    assert!(count > 0);
    assert!(!full_text.is_empty());
}
