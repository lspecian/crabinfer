//! Tool Framework — define, register, and execute tools for the AI agent.
//!
//! Tools are functions the agent can call to interact with the environment
//! (read files, execute commands, fetch URLs, etc.). Each tool has a name,
//! description, JSON Schema parameters, and an execute function.

use crate::CrabInferError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Result of executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    pub output: String,
    pub is_error: bool,
}

/// A tool call parsed from the model's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Trait for implementing tools the agent can use.
pub trait Tool: Send + Sync {
    /// Unique tool name (e.g., "file_read", "shell_exec").
    fn name(&self) -> &str;

    /// Human-readable description of what the tool does.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's parameters.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given arguments.
    fn execute(&self, args: Value) -> Result<String, CrabInferError>;
}

/// Registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Execute a tool call.
    pub fn execute(&self, call: &ToolCall) -> ToolResult {
        match self.tools.get(&call.name) {
            Some(tool) => match tool.execute(call.arguments.clone()) {
                Ok(output) => ToolResult {
                    tool_name: call.name.clone(),
                    output,
                    is_error: false,
                },
                Err(e) => ToolResult {
                    tool_name: call.name.clone(),
                    output: format!("Error: {}", e),
                    is_error: true,
                },
            },
            None => ToolResult {
                tool_name: call.name.clone(),
                output: format!("Unknown tool: {}", call.name),
                is_error: true,
            },
        }
    }

    /// List all registered tool names.
    pub fn tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Generate tool descriptions for injection into the system prompt.
    pub fn tools_prompt(&self) -> String {
        let mut sections = Vec::new();
        let mut names: Vec<&String> = self.tools.keys().collect();
        names.sort();

        for name in names {
            let tool = &self.tools[name];
            let schema = serde_json::to_string_pretty(&tool.parameters_schema())
                .unwrap_or_default();
            sections.push(format!(
                "### {}\n{}\nParameters:\n```json\n{}\n```",
                tool.name(),
                tool.description(),
                schema
            ));
        }

        if sections.is_empty() {
            return String::new();
        }

        format!(
            "You have access to the following tools. To use a tool, respond with a tool call in this exact format:\n\
            <tool_call>{{\"name\": \"tool_name\", \"arguments\": {{...}}}}</tool_call>\n\n\
            You can make multiple tool calls in one response. After each tool call, you will receive the result.\n\
            When you have enough information, respond with a normal text message (no tool call).\n\n\
            Available tools:\n\n{}",
            sections.join("\n\n")
        )
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Built-in Tools ──────────────────────────────────────────────────────────

/// Read a file's contents.
pub struct FileReadTool;

impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at the given path."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path to read"
                }
            },
            "required": ["path"]
        })
    }

    fn execute(&self, args: Value) -> Result<String, CrabInferError> {
        let path = args["path"]
            .as_str()
            .ok_or(CrabInferError::InvalidConfig)?;

        let content = std::fs::read_to_string(path).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to read '{}': {}", path, e),
        })?;

        // Truncate very large files
        if content.len() > 50_000 {
            Ok(format!(
                "{}\n\n[... truncated, {} total bytes]",
                &content[..50_000],
                content.len()
            ))
        } else {
            Ok(content)
        }
    }
}

/// Write content to a file.
pub struct FileWriteTool;

impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "file_write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    fn execute(&self, args: Value) -> Result<String, CrabInferError> {
        let path = args["path"]
            .as_str()
            .ok_or(CrabInferError::InvalidConfig)?;
        let content = args["content"]
            .as_str()
            .ok_or(CrabInferError::InvalidConfig)?;

        // Create parent directories
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| CrabInferError::StorageError {
                reason: format!("Failed to create directories: {}", e),
            })?;
        }

        std::fs::write(path, content).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to write '{}': {}", path, e),
        })?;

        Ok(format!("Wrote {} bytes to {}", content.len(), path))
    }
}

/// List files in a directory.
pub struct FileListTool;

impl Tool for FileListTool {
    fn name(&self) -> &str {
        "file_list"
    }

    fn description(&self) -> &str {
        "List files and directories in the given path."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list"
                }
            },
            "required": ["path"]
        })
    }

    fn execute(&self, args: Value) -> Result<String, CrabInferError> {
        let path = args["path"]
            .as_str()
            .ok_or(CrabInferError::InvalidConfig)?;

        let entries =
            std::fs::read_dir(path).map_err(|e| CrabInferError::StorageError {
                reason: format!("Failed to list '{}': {}", path, e),
            })?;

        let mut lines = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| CrabInferError::StorageError {
                reason: format!("Failed to read entry: {}", e),
            })?;
            let file_type = entry.file_type().map_err(|e| CrabInferError::StorageError {
                reason: format!("Failed to get file type: {}", e),
            })?;
            let prefix = if file_type.is_dir() { "d " } else { "  " };
            let name = entry.file_name().to_string_lossy().to_string();
            lines.push(format!("{}{}", prefix, name));
        }

        lines.sort();
        Ok(lines.join("\n"))
    }
}

/// Execute a shell command (sandboxed: no destructive commands).
pub struct ShellExecTool {
    /// Working directory for commands.
    pub working_dir: Option<String>,
}

impl ShellExecTool {
    pub fn new() -> Self {
        Self { working_dir: None }
    }

    pub fn with_working_dir(mut self, dir: &str) -> Self {
        self.working_dir = Some(dir.to_string());
        self
    }
}

impl Default for ShellExecTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Commands that are blocked for safety.
const BLOCKED_COMMANDS: &[&str] = &[
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){",
    "fork bomb",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "kill -9 1",
];

impl Tool for ShellExecTool {
    fn name(&self) -> &str {
        "shell_exec"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output. Destructive commands are blocked."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: Value) -> Result<String, CrabInferError> {
        let command = args["command"]
            .as_str()
            .ok_or(CrabInferError::InvalidConfig)?;

        // Safety check
        let lower = command.to_lowercase();
        for blocked in BLOCKED_COMMANDS {
            if lower.contains(blocked) {
                return Err(CrabInferError::InvalidConfig);
            }
        }

        let mut cmd = std::process::Command::new("sh");
        cmd.arg("-c").arg(command);

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        let output = cmd.output().map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to execute command: {}", e),
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("stderr: ");
            result.push_str(&stderr);
        }

        if !output.status.success() {
            result = format!(
                "Command exited with code {}\n{}",
                output.status.code().unwrap_or(-1),
                result
            );
        }

        // Truncate long output
        if result.len() > 30_000 {
            result.truncate(30_000);
            result.push_str("\n[... output truncated]");
        }

        Ok(result)
    }
}

/// Fetch content from a URL.
#[cfg(feature = "providers")]
pub struct WebFetchTool;

#[cfg(feature = "providers")]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch the text content of a URL. Returns the response body."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                }
            },
            "required": ["url"]
        })
    }

    fn execute(&self, args: Value) -> Result<String, CrabInferError> {
        let url = args["url"]
            .as_str()
            .ok_or(CrabInferError::InvalidConfig)?;

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| CrabInferError::NetworkError {
                reason: format!("Failed to create HTTP client: {}", e),
            })?;

        let response = client.get(url).send().map_err(|e| CrabInferError::NetworkError {
            reason: format!("Failed to fetch '{}': {}", url, e),
        })?;

        let status = response.status();
        let body = response.text().map_err(|e| CrabInferError::NetworkError {
            reason: format!("Failed to read response: {}", e),
        })?;

        if !status.is_success() {
            return Ok(format!("HTTP {} — {}", status, &body[..body.len().min(2000)]));
        }

        // Truncate large responses
        if body.len() > 30_000 {
            Ok(format!(
                "{}\n[... truncated, {} total bytes]",
                &body[..30_000],
                body.len()
            ))
        } else {
            Ok(body)
        }
    }
}

/// Create a tool registry with all built-in tools.
pub fn default_tools() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(FileReadTool));
    registry.register(Arc::new(FileWriteTool));
    registry.register(Arc::new(FileListTool));
    registry.register(Arc::new(ShellExecTool::default()));
    #[cfg(feature = "providers")]
    registry.register(Arc::new(WebFetchTool));
    registry
}

// ─── Tool Call Parsing ───────────────────────────────────────────────────────

/// Parse tool calls from model output.
///
/// Looks for patterns like:
/// `<tool_call>{"name": "file_read", "arguments": {"path": "/tmp/foo"}}</tool_call>`
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut search_start = 0;

    while let Some(start) = text[search_start..].find("<tool_call>") {
        let abs_start = search_start + start + "<tool_call>".len();
        if let Some(end) = text[abs_start..].find("</tool_call>") {
            let json_str = text[abs_start..abs_start + end].trim();
            if let Ok(call) = serde_json::from_str::<ToolCall>(json_str) {
                calls.push(call);
            }
            search_start = abs_start + end + "</tool_call>".len();
        } else {
            break;
        }
    }

    calls
}

/// Check if the model's response contains any tool calls.
pub fn has_tool_calls(text: &str) -> bool {
    text.contains("<tool_call>")
}

/// Extract the text portion of a response (everything outside tool_call tags).
pub fn extract_text(text: &str) -> String {
    let mut result = text.to_string();
    // Remove all <tool_call>...</tool_call> blocks
    while let Some(start) = result.find("<tool_call>") {
        if let Some(end) = result[start..].find("</tool_call>") {
            let remove_end = start + end + "</tool_call>".len();
            result = format!("{}{}", &result[..start], &result[remove_end..]);
        } else {
            break;
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_calls() {
        let text = r#"I'll read that file for you.
<tool_call>{"name": "file_read", "arguments": {"path": "/tmp/test.txt"}}</tool_call>
Let me also check the directory.
<tool_call>{"name": "file_list", "arguments": {"path": "/tmp"}}</tool_call>"#;

        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "file_read");
        assert_eq!(calls[1].name, "file_list");
    }

    #[test]
    fn test_has_tool_calls() {
        assert!(has_tool_calls("text <tool_call>{}</tool_call> more"));
        assert!(!has_tool_calls("just plain text"));
    }

    #[test]
    fn test_extract_text() {
        let text = "Hello <tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call> World";
        assert_eq!(extract_text(text), "Hello  World");
    }

    #[test]
    fn test_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(FileReadTool));
        registry.register(Arc::new(FileListTool));

        assert_eq!(registry.len(), 2);
        assert!(registry.get("file_read").is_some());
        assert!(registry.get("nonexistent").is_none());

        let prompt = registry.tools_prompt();
        assert!(prompt.contains("file_read"));
        assert!(prompt.contains("file_list"));
    }

    #[test]
    fn test_file_list_tool() {
        let tool = FileListTool;
        let result = tool.execute(serde_json::json!({"path": "/tmp"}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_shell_blocked_command() {
        let tool = ShellExecTool::default();
        let result = tool.execute(serde_json::json!({"command": "rm -rf /"}));
        assert!(result.is_err());
    }

    #[test]
    fn test_shell_safe_command() {
        let tool = ShellExecTool::default();
        let result = tool.execute(serde_json::json!({"command": "echo hello"}));
        assert!(result.is_ok());
        assert!(result.unwrap().contains("hello"));
    }

    #[test]
    fn test_default_tools() {
        let registry = default_tools();
        assert!(registry.len() >= 4); // file_read, file_write, file_list, shell_exec
    }
}
