//! Node.js bindings for A8 (Agent + Tool framework).
//!
//! Exposes Agent, ToolRegistry, and tool call parsing to JS.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::to_napi_error;

// ─── Tool Call Types ────────────────────────────────────────────────────────

/// A parsed tool call from model output.
#[napi(object)]
pub struct JsToolCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// Result of executing a tool.
#[napi(object)]
pub struct JsToolResult {
    pub tool_name: String,
    pub output: String,
    pub is_error: bool,
}

/// Record of a tool execution during an agent turn.
#[napi(object)]
pub struct JsAgentToolExecution {
    pub tool_name: String,
    pub arguments: String, // JSON string
    pub output: String,
    pub is_error: bool,
}

/// Agent response from a single turn.
#[napi(object)]
pub struct JsAgentResponse {
    pub text: String,
    pub tool_calls: Vec<JsAgentToolExecution>,
    pub rounds: u32,
}

// ─── Tool Call Parsing (standalone functions) ──────────────────────────────

/// Parse tool calls from model output text.
///
/// Looks for `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` patterns.
#[napi]
pub fn parse_tool_calls(text: String) -> Vec<JsToolCall> {
    crabinfer_core::tools::parse_tool_calls(&text)
        .into_iter()
        .map(|tc| JsToolCall {
            name: tc.name,
            arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
        })
        .collect()
}

/// Check if text contains tool calls.
#[napi]
pub fn has_tool_calls(text: String) -> bool {
    crabinfer_core::tools::has_tool_calls(&text)
}

/// Extract plain text from a response (removing tool_call blocks).
#[napi]
pub fn extract_text_from_response(text: String) -> String {
    crabinfer_core::tools::extract_text(&text)
}

// ─── ToolRegistry ──────────────────────────────────────────────────────────

/// Registry of tools that can be used by the agent.
#[napi]
pub struct ToolRegistry {
    inner: crabinfer_core::tools::ToolRegistry,
}

#[napi]
impl ToolRegistry {
    /// Create a tool registry with all built-in tools (file_read, file_write, file_list, shell_exec, web_fetch).
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: crabinfer_core::tools::default_tools(),
        }
    }

    /// Get all registered tool names.
    #[napi]
    pub fn tool_names(&self) -> Vec<String> {
        self.inner.tool_names()
    }

    /// Generate the tools section for a system prompt.
    #[napi]
    pub fn tools_prompt(&self) -> String {
        self.inner.tools_prompt()
    }

    /// Number of registered tools.
    #[napi(getter)]
    pub fn count(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Execute a tool call by name with JSON arguments.
    #[napi]
    pub fn execute(&self, name: String, arguments_json: String) -> JsToolResult {
        let arguments: serde_json::Value =
            serde_json::from_str(&arguments_json).unwrap_or(serde_json::Value::Null);
        let call = crabinfer_core::tools::ToolCall {
            name,
            arguments,
        };
        let result = self.inner.execute(&call);
        JsToolResult {
            tool_name: result.tool_name,
            output: result.output,
            is_error: result.is_error,
        }
    }

    /// Connect to an MCP stdio server and register its tools.
    /// Returns the number of tools registered.
    #[napi]
    pub fn connect_mcp_stdio(&mut self, command: String, args: Vec<String>) -> Result<u32> {
        let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let client = crabinfer_core::mcp::McpStdioClient::connect(&command, &str_args)
            .map_err(to_napi_error)?;
        let client = std::sync::Arc::new(client);
        let count = crabinfer_core::mcp::register_mcp_tools(&mut self.inner, client)
            .map_err(to_napi_error)?;
        Ok(count as u32)
    }

    /// Connect to an MCP HTTP server and register its tools.
    /// Returns the number of tools registered.
    #[cfg(feature = "providers")]
    #[napi]
    pub fn connect_mcp_http(&mut self, url: String) -> Result<u32> {
        let client = crabinfer_core::mcp::McpHttpClient::connect(&url)
            .map_err(to_napi_error)?;
        let client = std::sync::Arc::new(client);
        let count = crabinfer_core::mcp::register_mcp_tools(&mut self.inner, client)
            .map_err(to_napi_error)?;
        Ok(count as u32)
    }
}

// ─── McpServerConfig ──────────────────────────────────────────────────────

/// MCP server configuration for JS.
#[napi(object)]
pub struct JsMcpServerConfig {
    pub name: String,
    /// "stdio" or "http"
    pub transport: String,
    /// Command (stdio) or URL (http)
    pub command: String,
    pub args: Vec<String>,
    pub enabled: bool,
    pub description: String,
}

/// MCP tool info for JS.
#[napi(object)]
pub struct JsMcpToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: String, // JSON string
}

// ─── McpServerRegistry ──────────────────────────────────────────────────────

/// Registry of configured MCP servers.
#[napi]
pub struct McpServerRegistry {
    inner: crabinfer_core::mcp::McpServerRegistry,
}

#[napi]
impl McpServerRegistry {
    /// Create an empty registry.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: crabinfer_core::mcp::McpServerRegistry::new(),
        }
    }

    /// Load from a file path.
    #[napi(factory)]
    pub fn load(path: String) -> Self {
        Self {
            inner: crabinfer_core::mcp::McpServerRegistry::load(&path),
        }
    }

    /// Load from the default location (~/.crabinfer/mcp-servers.json).
    #[napi(factory)]
    pub fn load_default() -> Self {
        Self {
            inner: crabinfer_core::mcp::McpServerRegistry::load_default(),
        }
    }

    /// Save to disk.
    #[napi]
    pub fn save(&self) -> Result<()> {
        self.inner.save().map_err(to_napi_error)
    }

    /// Set the file path for persistence.
    #[napi]
    pub fn with_persist_path(&mut self, path: String) {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.with_persist_path(&path);
    }

    /// Add a server configuration.
    #[napi]
    pub fn add(&mut self, config: JsMcpServerConfig) {
        let transport = match config.transport.as_str() {
            "http" => crabinfer_core::mcp::McpTransport::Http,
            _ => crabinfer_core::mcp::McpTransport::Stdio,
        };
        self.inner.add(crabinfer_core::mcp::McpServerConfig {
            name: config.name,
            transport,
            command: config.command,
            args: config.args,
            enabled: config.enabled,
            description: config.description,
        });
    }

    /// Remove a server by name.
    #[napi]
    pub fn remove(&mut self, name: String) -> bool {
        self.inner.remove(&name)
    }

    /// Enable or disable a server.
    #[napi]
    pub fn set_enabled(&mut self, name: String, enabled: bool) -> bool {
        self.inner.set_enabled(&name, enabled)
    }

    /// Get all server configs.
    #[napi]
    pub fn servers(&self) -> Vec<JsMcpServerConfig> {
        self.inner
            .servers()
            .iter()
            .map(|s| JsMcpServerConfig {
                name: s.name.clone(),
                transport: match s.transport {
                    crabinfer_core::mcp::McpTransport::Stdio => "stdio".to_string(),
                    crabinfer_core::mcp::McpTransport::Http => "http".to_string(),
                },
                command: s.command.clone(),
                args: s.args.clone(),
                enabled: s.enabled,
                description: s.description.clone(),
            })
            .collect()
    }

    /// Number of configured servers.
    #[napi(getter)]
    pub fn count(&self) -> u32 {
        self.inner.count() as u32
    }
}
