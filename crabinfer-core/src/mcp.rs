//! MCP (Model Context Protocol) — client and server implementation.
//!
//! Supports two transport modes:
//! - **Stdio**: spawn a child process, communicate via stdin/stdout (local MCP servers)
//! - **HTTP**: connect to a remote MCP server via HTTP + JSON-RPC (remote/network servers)
//!
//! Also provides:
//! - `McpServerConfig` / `McpServerRegistry` for server discovery and management
//! - `McpServer` for exposing CrabInfer's tools as an MCP server

use crate::tools::{Tool, ToolRegistry};
use crate::CrabInferError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

// ─── JSON-RPC 2.0 Types ─────────────────────────────────────────────────────

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    #[allow(dead_code)]
    id: Option<u64>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i64,
    message: String,
}

// ─── MCP Tool Info ───────────────────────────────────────────────────────────

/// Description of a tool available from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

// ─── MCP Client Trait ────────────────────────────────────────────────────────

/// Unified interface for MCP clients (stdio and HTTP).
pub trait McpClient: Send + Sync {
    /// Server name for display purposes.
    fn server_name(&self) -> &str;

    /// List available tools from the server.
    fn list_tools(&self) -> Result<Vec<McpToolInfo>, CrabInferError>;

    /// Call a tool on the server.
    fn call_tool(&self, name: &str, arguments: Value) -> Result<String, CrabInferError>;
}

// ─── MCP Stdio Client ───────────────────────────────────────────────────────

/// MCP client that communicates with a server over stdio (JSON-RPC 2.0).
pub struct McpStdioClient {
    process: Mutex<Child>,
    request_id: Mutex<u64>,
    server_name: String,
}

impl McpStdioClient {
    /// Spawn an MCP server process and connect via stdio.
    ///
    /// `command` is the executable and `args` are command-line arguments.
    /// Example: `McpStdioClient::connect("npx", &["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])`
    pub fn connect(command: &str, args: &[&str]) -> Result<Self, CrabInferError> {
        let child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| CrabInferError::NetworkError {
                reason: format!("Failed to spawn MCP server '{}': {}", command, e),
            })?;

        let server_name = format!("mcp-stdio:{}", command);
        let client = Self {
            process: Mutex::new(child),
            request_id: Mutex::new(1),
            server_name,
        };

        client.initialize()?;
        Ok(client)
    }

    fn send_request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, CrabInferError> {
        let id = {
            let mut id = self.request_id.lock().unwrap();
            let current = *id;
            *id += 1;
            current
        };

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };

        let request_json =
            serde_json::to_string(&request).map_err(|e| CrabInferError::NetworkError {
                reason: format!("Failed to serialize request: {}", e),
            })?;

        let mut process = self.process.lock().unwrap();

        // Write request
        if let Some(ref mut stdin) = process.stdin {
            writeln!(stdin, "{}", request_json).map_err(|e| CrabInferError::NetworkError {
                reason: format!("Failed to write to MCP server: {}", e),
            })?;
            stdin.flush().map_err(|e| CrabInferError::NetworkError {
                reason: format!("Failed to flush: {}", e),
            })?;
        } else {
            return Err(CrabInferError::NetworkError {
                reason: "MCP server stdin not available".to_string(),
            });
        }

        // Read response
        if let Some(ref mut stdout) = process.stdout {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            reader
                .read_line(&mut line)
                .map_err(|e| CrabInferError::NetworkError {
                    reason: format!("Failed to read from MCP server: {}", e),
                })?;

            let response: JsonRpcResponse =
                serde_json::from_str(&line).map_err(|e| CrabInferError::NetworkError {
                    reason: format!("Invalid JSON-RPC response: {}", e),
                })?;

            if let Some(error) = response.error {
                return Err(CrabInferError::NetworkError {
                    reason: format!("MCP error: {}", error.message),
                });
            }

            Ok(response.result.unwrap_or(Value::Null))
        } else {
            Err(CrabInferError::NetworkError {
                reason: "MCP server stdout not available".to_string(),
            })
        }
    }

    fn initialize(&self) -> Result<(), CrabInferError> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "crabinfer",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        self.send_request("initialize", Some(params))?;
        Ok(())
    }
}

impl McpClient for McpStdioClient {
    fn server_name(&self) -> &str {
        &self.server_name
    }

    fn list_tools(&self) -> Result<Vec<McpToolInfo>, CrabInferError> {
        let result = self.send_request("tools/list", None)?;
        parse_tools_list(&result)
    }

    fn call_tool(&self, name: &str, arguments: Value) -> Result<String, CrabInferError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });

        let result = self.send_request("tools/call", Some(params))?;
        extract_tool_result(&result)
    }
}

impl Drop for McpStdioClient {
    fn drop(&mut self) {
        if let Ok(mut process) = self.process.lock() {
            let _ = process.kill();
        }
    }
}

// ─── MCP HTTP Client ─────────────────────────────────────────────────────────

/// MCP client that communicates with a server over HTTP (JSON-RPC 2.0).
///
/// The HTTP transport sends POST requests to the server's endpoint.
/// Each request contains a JSON-RPC 2.0 message; the response body
/// contains the JSON-RPC 2.0 response.
#[cfg(feature = "providers")]
pub struct McpHttpClient {
    base_url: String,
    client: reqwest::blocking::Client,
    request_id: Mutex<u64>,
    server_name: String,
}

#[cfg(feature = "providers")]
impl McpHttpClient {
    /// Connect to an MCP server over HTTP.
    ///
    /// `base_url` should be the server's MCP endpoint (e.g. `http://localhost:3000/mcp`).
    pub fn connect(base_url: &str) -> Result<Self, CrabInferError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| CrabInferError::NetworkError {
                reason: format!("Failed to create HTTP client: {}", e),
            })?;

        let server_name = format!("mcp-http:{}", base_url);
        let instance = Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
            request_id: Mutex::new(1),
            server_name,
        };

        instance.initialize()?;
        Ok(instance)
    }

    fn send_request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, CrabInferError> {
        let id = {
            let mut id = self.request_id.lock().unwrap();
            let current = *id;
            *id += 1;
            current
        };

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };

        let response = self
            .client
            .post(&self.base_url)
            .json(&request)
            .send()
            .map_err(|e| CrabInferError::NetworkError {
                reason: format!("HTTP request to MCP server failed: {}", e),
            })?;

        if !response.status().is_success() {
            return Err(CrabInferError::NetworkError {
                reason: format!("MCP server returned HTTP {}", response.status()),
            });
        }

        let rpc_response: JsonRpcResponse =
            response.json().map_err(|e| CrabInferError::NetworkError {
                reason: format!("Invalid JSON-RPC response: {}", e),
            })?;

        if let Some(error) = rpc_response.error {
            return Err(CrabInferError::NetworkError {
                reason: format!("MCP error: {}", error.message),
            });
        }

        Ok(rpc_response.result.unwrap_or(Value::Null))
    }

    fn initialize(&self) -> Result<(), CrabInferError> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "crabinfer",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        self.send_request("initialize", Some(params))?;
        Ok(())
    }
}

#[cfg(feature = "providers")]
impl McpClient for McpHttpClient {
    fn server_name(&self) -> &str {
        &self.server_name
    }

    fn list_tools(&self) -> Result<Vec<McpToolInfo>, CrabInferError> {
        let result = self.send_request("tools/list", None)?;
        parse_tools_list(&result)
    }

    fn call_tool(&self, name: &str, arguments: Value) -> Result<String, CrabInferError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });
        let result = self.send_request("tools/call", Some(params))?;
        extract_tool_result(&result)
    }
}

// ─── Shared Helpers ──────────────────────────────────────────────────────────

fn parse_tools_list(result: &Value) -> Result<Vec<McpToolInfo>, CrabInferError> {
    let tools = result["tools"]
        .as_array()
        .ok_or(CrabInferError::NetworkError {
            reason: "Invalid tools/list response".to_string(),
        })?;

    let mut tool_infos = Vec::new();
    for tool in tools {
        tool_infos.push(McpToolInfo {
            name: tool["name"].as_str().unwrap_or("").to_string(),
            description: tool["description"].as_str().unwrap_or("").to_string(),
            input_schema: tool["inputSchema"].clone(),
        });
    }

    Ok(tool_infos)
}

fn extract_tool_result(result: &Value) -> Result<String, CrabInferError> {
    if let Some(content) = result["content"].as_array() {
        let texts: Vec<String> = content
            .iter()
            .filter_map(|item| {
                if item["type"].as_str() == Some("text") {
                    item["text"].as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect();
        Ok(texts.join("\n"))
    } else {
        Ok(serde_json::to_string_pretty(result).unwrap_or_default())
    }
}

// ─── MCP Tool Wrapper ────────────────────────────────────────────────────────

/// Wraps an MCP server tool as a crabinfer Tool for use in the agent.
struct McpTool {
    client: Arc<dyn McpClient>,
    info: McpToolInfo,
}

impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.info.name
    }

    fn description(&self) -> &str {
        &self.info.description
    }

    fn parameters_schema(&self) -> Value {
        self.info.input_schema.clone()
    }

    fn execute(&self, args: Value) -> Result<String, CrabInferError> {
        self.client.call_tool(&self.info.name, args)
    }
}

/// Register all tools from an MCP client into a tool registry.
pub fn register_mcp_tools(
    registry: &mut ToolRegistry,
    client: Arc<dyn McpClient>,
) -> Result<usize, CrabInferError> {
    let tools = client.list_tools()?;
    let count = tools.len();

    for info in tools {
        let tool = McpTool {
            client: client.clone(),
            info,
        };
        registry.register(Arc::new(tool));
    }

    Ok(count)
}

// ─── MCP Server Config & Registry ───────────────────────────────────────────

/// Transport type for an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    Stdio,
    Http,
}

/// Configuration for a single MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Unique name for this server.
    pub name: String,
    /// Transport type.
    pub transport: McpTransport,
    /// For stdio: the command to execute. For HTTP: the base URL.
    pub command: String,
    /// For stdio: command-line arguments. Ignored for HTTP.
    #[serde(default)]
    pub args: Vec<String>,
    /// Whether this server is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional description.
    #[serde(default)]
    pub description: String,
}

fn default_true() -> bool {
    true
}

/// Registry of configured MCP servers, persisted to a JSON file.
///
/// Default location: `~/.crabinfer/mcp-servers.json`
pub struct McpServerRegistry {
    servers: Vec<McpServerConfig>,
    persist_path: Option<String>,
}

impl McpServerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            servers: Vec::new(),
            persist_path: None,
        }
    }

    /// Set the file path for persistence.
    pub fn with_persist_path(mut self, path: &str) -> Self {
        self.persist_path = Some(path.to_string());
        self
    }

    /// Load from a JSON file, or return empty if the file doesn't exist.
    pub fn load(path: &str) -> Self {
        let servers = std::fs::read_to_string(path)
            .ok()
            .and_then(|json| serde_json::from_str::<Vec<McpServerConfig>>(&json).ok())
            .unwrap_or_default();

        Self {
            servers,
            persist_path: Some(path.to_string()),
        }
    }

    /// Load from the default location (`~/.crabinfer/mcp-servers.json`).
    pub fn load_default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let path = format!("{}/.crabinfer/mcp-servers.json", home);
        Self::load(&path)
    }

    /// Save to disk.
    pub fn save(&self) -> Result<(), CrabInferError> {
        let path = self.persist_path.as_ref().ok_or(CrabInferError::StorageError {
            reason: "No persist path set".to_string(),
        })?;

        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let json = serde_json::to_string_pretty(&self.servers).map_err(|e| {
            CrabInferError::StorageError {
                reason: format!("Serialization failed: {}", e),
            }
        })?;

        std::fs::write(path, json).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to write: {}", e),
        })
    }

    /// Add a server configuration.
    pub fn add(&mut self, config: McpServerConfig) {
        // Replace if name exists
        self.servers.retain(|s| s.name != config.name);
        self.servers.push(config);
    }

    /// Remove a server by name. Returns true if it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.servers.len();
        self.servers.retain(|s| s.name != name);
        self.servers.len() < before
    }

    /// Enable or disable a server.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if let Some(server) = self.servers.iter_mut().find(|s| s.name == name) {
            server.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Get all server configs.
    pub fn servers(&self) -> &[McpServerConfig] {
        &self.servers
    }

    /// Get only enabled servers.
    pub fn enabled_servers(&self) -> Vec<&McpServerConfig> {
        self.servers.iter().filter(|s| s.enabled).collect()
    }

    /// Get a server by name.
    pub fn get(&self, name: &str) -> Option<&McpServerConfig> {
        self.servers.iter().find(|s| s.name == name)
    }

    /// Number of configured servers.
    pub fn count(&self) -> usize {
        self.servers.len()
    }

    /// Connect to all enabled servers and register their tools.
    ///
    /// Returns a list of (server_name, tool_count) for successfully connected servers.
    /// Servers that fail to connect are skipped with a warning printed to stderr.
    pub fn connect_all(
        &self,
        registry: &mut ToolRegistry,
    ) -> Vec<(String, usize)> {
        let mut connected = Vec::new();

        for config in self.enabled_servers() {
            match connect_server(config) {
                Ok(client) => {
                    match register_mcp_tools(registry, client) {
                        Ok(count) => {
                            connected.push((config.name.clone(), count));
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: failed to list tools from '{}': {}",
                                config.name, e
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: failed to connect to MCP server '{}': {}",
                        config.name, e
                    );
                }
            }
        }

        connected
    }
}

impl Default for McpServerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Connect to a single MCP server based on its config.
pub fn connect_server(
    config: &McpServerConfig,
) -> Result<Arc<dyn McpClient>, CrabInferError> {
    match config.transport {
        McpTransport::Stdio => {
            let args: Vec<&str> = config.args.iter().map(|s| s.as_str()).collect();
            let client = McpStdioClient::connect(&config.command, &args)?;
            Ok(Arc::new(client))
        }
        McpTransport::Http => {
            #[cfg(feature = "providers")]
            {
                let client = McpHttpClient::connect(&config.command)?;
                Ok(Arc::new(client))
            }
            #[cfg(not(feature = "providers"))]
            {
                Err(CrabInferError::NetworkError {
                    reason: "HTTP MCP transport requires the 'providers' feature".to_string(),
                })
            }
        }
    }
}

// ─── Built-in MCP Server ─────────────────────────────────────────────────────

/// A simple MCP server that exposes CrabInfer's built-in tools via JSON-RPC 2.0 over stdio.
///
/// This allows external clients (Claude Desktop, other agents) to use CrabInfer's
/// file, shell, and web tools via the MCP protocol.
///
/// Usage: run as a child process, reads JSON-RPC from stdin, writes to stdout.
pub struct McpServer {
    tools: ToolRegistry,
    server_info: Value,
}

impl McpServer {
    /// Create a new MCP server exposing the default built-in tools.
    pub fn new() -> Self {
        Self {
            tools: crate::tools::default_tools(),
            server_info: serde_json::json!({
                "name": "crabinfer",
                "version": env!("CARGO_PKG_VERSION")
            }),
        }
    }

    /// Create a server with a custom tool registry.
    pub fn with_tools(tools: ToolRegistry) -> Self {
        Self {
            tools,
            server_info: serde_json::json!({
                "name": "crabinfer",
                "version": env!("CARGO_PKG_VERSION")
            }),
        }
    }

    /// Run the server, reading from stdin and writing to stdout.
    ///
    /// This blocks forever (until stdin is closed or the process is killed).
    pub fn run(&self) -> Result<(), CrabInferError> {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let reader = BufReader::new(stdin.lock());

        for line in reader.lines() {
            let line = line.map_err(|e| CrabInferError::NetworkError {
                reason: format!("Failed to read stdin: {}", e),
            })?;

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let request: Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    let error_response = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": {
                            "code": -32700,
                            "message": format!("Parse error: {}", e)
                        }
                    });
                    writeln!(stdout.lock(), "{}", error_response).ok();
                    stdout.lock().flush().ok();
                    continue;
                }
            };

            let id = request.get("id").cloned();
            let method = request["method"].as_str().unwrap_or("");

            let result = self.handle_method(method, request.get("params"));

            let response = match result {
                Ok(value) => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": value
                }),
                Err(e) => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32603,
                        "message": format!("{}", e)
                    }
                }),
            };

            writeln!(stdout.lock(), "{}", response).map_err(|e| {
                CrabInferError::NetworkError {
                    reason: format!("Failed to write stdout: {}", e),
                }
            })?;
            stdout.lock().flush().ok();
        }

        Ok(())
    }

    fn handle_method(
        &self,
        method: &str,
        params: Option<&Value>,
    ) -> Result<Value, CrabInferError> {
        match method {
            "initialize" => Ok(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": self.server_info
            })),

            "tools/list" => {
                let tool_list: Vec<Value> = self
                    .tools
                    .tool_names()
                    .iter()
                    .filter_map(|name| {
                        self.tools.get(name).map(|tool| {
                            serde_json::json!({
                                "name": tool.name(),
                                "description": tool.description(),
                                "inputSchema": tool.parameters_schema()
                            })
                        })
                    })
                    .collect();

                Ok(serde_json::json!({ "tools": tool_list }))
            }

            "tools/call" => {
                let params = params.ok_or(CrabInferError::InvalidConfig)?;
                let name = params["name"]
                    .as_str()
                    .ok_or(CrabInferError::InvalidConfig)?;
                let arguments = params["arguments"].clone();

                let call = crate::tools::ToolCall {
                    name: name.to_string(),
                    arguments,
                };
                let result = self.tools.execute(&call);

                Ok(serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": result.output
                    }],
                    "isError": result.is_error
                }))
            }

            _ => Err(CrabInferError::NetworkError {
                reason: format!("Unknown method: {}", method),
            }),
        }
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_tool_info_serialization() {
        let info = McpToolInfo {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("test_tool"));
    }

    #[test]
    fn test_server_config_serialization() {
        let config = McpServerConfig {
            name: "test-server".to_string(),
            transport: McpTransport::Stdio,
            command: "npx".to_string(),
            args: vec!["-y".to_string(), "@mcp/server-fs".to_string()],
            enabled: true,
            description: "File system server".to_string(),
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        assert!(json.contains("test-server"));
        assert!(json.contains("stdio"));

        let parsed: McpServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "test-server");
        assert_eq!(parsed.transport, McpTransport::Stdio);
    }

    #[test]
    fn test_server_config_http() {
        let config = McpServerConfig {
            name: "remote".to_string(),
            transport: McpTransport::Http,
            command: "http://localhost:3000/mcp".to_string(),
            args: vec![],
            enabled: true,
            description: "Remote server".to_string(),
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("http"));
    }

    #[test]
    fn test_server_registry() {
        let mut registry = McpServerRegistry::new();
        assert_eq!(registry.count(), 0);

        registry.add(McpServerConfig {
            name: "fs".to_string(),
            transport: McpTransport::Stdio,
            command: "npx".to_string(),
            args: vec![],
            enabled: true,
            description: String::new(),
        });

        registry.add(McpServerConfig {
            name: "web".to_string(),
            transport: McpTransport::Http,
            command: "http://localhost:3000".to_string(),
            args: vec![],
            enabled: false,
            description: String::new(),
        });

        assert_eq!(registry.count(), 2);
        assert_eq!(registry.enabled_servers().len(), 1);
        assert!(registry.get("fs").is_some());
        assert!(registry.get("missing").is_none());

        // Enable/disable
        registry.set_enabled("web", true);
        assert_eq!(registry.enabled_servers().len(), 2);

        // Remove
        assert!(registry.remove("fs"));
        assert_eq!(registry.count(), 1);
        assert!(!registry.remove("fs")); // already removed
    }

    #[test]
    fn test_server_registry_replace_on_add() {
        let mut registry = McpServerRegistry::new();
        registry.add(McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Stdio,
            command: "old-cmd".to_string(),
            args: vec![],
            enabled: true,
            description: String::new(),
        });

        registry.add(McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Http,
            command: "http://new-url".to_string(),
            args: vec![],
            enabled: true,
            description: String::new(),
        });

        assert_eq!(registry.count(), 1);
        assert_eq!(registry.get("test").unwrap().transport, McpTransport::Http);
    }

    #[test]
    fn test_server_registry_persistence() {
        let dir = std::env::temp_dir().join("crabinfer-mcp-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("servers.json");
        let path_str = path.to_str().unwrap();

        // Save
        let mut registry = McpServerRegistry::new().with_persist_path(path_str);
        registry.add(McpServerConfig {
            name: "test-server".to_string(),
            transport: McpTransport::Stdio,
            command: "echo".to_string(),
            args: vec!["hello".to_string()],
            enabled: true,
            description: "Test".to_string(),
        });
        registry.save().unwrap();

        // Load
        let loaded = McpServerRegistry::load(path_str);
        assert_eq!(loaded.count(), 1);
        assert_eq!(loaded.get("test-server").unwrap().command, "echo");

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_mcp_server_handles_initialize() {
        let server = McpServer::new();
        let result = server.handle_method("initialize", None).unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert!(result["serverInfo"]["name"].as_str().unwrap().contains("crabinfer"));
    }

    #[test]
    fn test_mcp_server_handles_tools_list() {
        let server = McpServer::new();
        let result = server.handle_method("tools/list", None).unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert!(tools.len() >= 4); // file_read, file_write, file_list, shell_exec
        assert!(tools.iter().any(|t| t["name"] == "file_read"));
    }

    #[test]
    fn test_mcp_server_handles_tools_call() {
        let server = McpServer::new();
        let params = serde_json::json!({
            "name": "file_list",
            "arguments": {"path": "/tmp"}
        });
        let result = server.handle_method("tools/call", Some(&params)).unwrap();
        assert!(result["content"].is_array());
        assert_eq!(result["isError"], false);
    }

    #[test]
    fn test_mcp_server_unknown_tool() {
        let server = McpServer::new();
        let params = serde_json::json!({
            "name": "nonexistent_tool",
            "arguments": {}
        });
        let result = server.handle_method("tools/call", Some(&params)).unwrap();
        assert_eq!(result["isError"], true);
    }

    #[test]
    fn test_mcp_server_unknown_method() {
        let server = McpServer::new();
        let result = server.handle_method("unknown/method", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_tools_list() {
        let value = serde_json::json!({
            "tools": [
                {
                    "name": "tool_a",
                    "description": "Tool A",
                    "inputSchema": {"type": "object"}
                },
                {
                    "name": "tool_b",
                    "description": "Tool B",
                    "inputSchema": {"type": "object"}
                }
            ]
        });
        let tools = parse_tools_list(&value).unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "tool_a");
        assert_eq!(tools[1].name, "tool_b");
    }
}
