//! Agent Runtime — autonomous loop with tool calling.
//!
//! The agent takes user input, sends it to the LLM (with tools, memory, and
//! knowledge context), parses tool calls from the response, executes them,
//! feeds results back, and repeats until the model produces a text-only response.
//!
//! ```text
//! User Input
//!   → Build prompt (system + tools + memory + knowledge + messages)
//!   → LLM generates response
//!   → Parse for tool calls
//!     → If tool calls: execute tools, add results to messages, loop back
//!     → If text only: return response to user
//! ```

use crate::conversation::ConversationMemory;
use crate::facts::MemoryStore;
use crate::knowledge::KnowledgeBase;
use crate::prompt::SystemPrompt;
use crate::provider::{CompletionRequest, Provider};
use crate::tools::{self, ToolRegistry};
use crate::CrabInferError;
use std::sync::Arc;

/// Configuration for the agent.
pub struct AgentConfig {
    /// Maximum number of tool-calling rounds per turn (prevents infinite loops).
    pub max_tool_rounds: u32,
    /// Maximum tokens for each LLM call.
    pub max_tokens: u32,
    /// Temperature for generation.
    pub temperature: f32,
    /// Top-p sampling.
    pub top_p: f32,
    /// Number of RAG results to retrieve per query.
    pub rag_top_k: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 10,
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            rag_top_k: 3,
        }
    }
}

/// Result of a single agent turn.
pub struct AgentResponse {
    /// The final text response from the agent.
    pub text: String,
    /// Tool calls that were executed during this turn.
    pub tool_calls: Vec<AgentToolExecution>,
    /// Number of LLM rounds needed.
    pub rounds: u32,
}

/// Record of a tool execution.
pub struct AgentToolExecution {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub output: String,
    pub is_error: bool,
}

/// The AI agent — combines LLM, tools, memory, and knowledge.
pub struct Agent {
    /// The LLM provider (local, cloud, or router).
    provider: Arc<dyn Provider>,
    /// Tool registry (built-in + MCP tools).
    tools: ToolRegistry,
    /// System prompt configuration.
    system_prompt: SystemPrompt,
    /// Conversation history.
    conversation: ConversationMemory,
    /// Persistent user facts.
    facts: MemoryStore,
    /// Optional knowledge base for RAG.
    knowledge: Option<KnowledgeBase>,
    /// Agent configuration.
    config: AgentConfig,
}

impl Agent {
    /// Create a new agent with the given provider.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider,
            tools: tools::default_tools(),
            system_prompt: SystemPrompt::conversational(),
            conversation: ConversationMemory::new("agent-default"),
            facts: MemoryStore::new(),
            knowledge: None,
            config: AgentConfig::default(),
        }
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, prompt: SystemPrompt) -> Self {
        self.system_prompt = prompt;
        self
    }

    /// Set the conversation memory.
    pub fn with_conversation(mut self, conv: ConversationMemory) -> Self {
        self.conversation = conv;
        self
    }

    /// Set the persistent fact store.
    pub fn with_facts(mut self, facts: MemoryStore) -> Self {
        self.facts = facts;
        self
    }

    /// Set the knowledge base for RAG.
    pub fn with_knowledge(mut self, kb: KnowledgeBase) -> Self {
        self.knowledge = Some(kb);
        self
    }

    /// Set the agent configuration.
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Get a mutable reference to the tool registry for adding custom tools.
    pub fn tools_mut(&mut self) -> &mut ToolRegistry {
        &mut self.tools
    }

    /// Get a reference to the conversation memory.
    pub fn conversation(&self) -> &ConversationMemory {
        &self.conversation
    }

    /// Get a mutable reference to the conversation memory.
    pub fn conversation_mut(&mut self) -> &mut ConversationMemory {
        &mut self.conversation
    }

    /// Get a mutable reference to the facts store.
    pub fn facts_mut(&mut self) -> &mut MemoryStore {
        &mut self.facts
    }

    /// Process a user message and return the agent's response.
    ///
    /// This runs the full agent loop:
    /// 1. Add user message to conversation
    /// 2. Build system prompt with tools + facts + knowledge
    /// 3. Send to LLM
    /// 4. Parse tool calls → execute → feed back → repeat
    /// 5. Return final text response
    pub fn run(&mut self, user_input: &str) -> Result<AgentResponse, CrabInferError> {
        // Add user message to conversation
        self.conversation.add_user_message(user_input);

        let mut tool_executions = Vec::new();
        let mut rounds = 0;

        loop {
            rounds += 1;
            if rounds > self.config.max_tool_rounds {
                // Safety limit: break out of potential infinite loops
                let msg = "I've reached the maximum number of tool calls for this turn. Let me summarize what I've found so far.".to_string();
                self.conversation.add_assistant_message(&msg);
                return Ok(AgentResponse {
                    text: msg,
                    tool_calls: tool_executions,
                    rounds,
                });
            }

            // Build the system prompt
            let facts_context = self.facts.as_prompt_context();
            let knowledge_context = self.get_knowledge_context(user_input);
            let tools_section = self.tools.tools_prompt();

            let mut system_parts = Vec::new();
            let base_prompt = self.system_prompt.build_with_context(&facts_context, &knowledge_context);
            if !base_prompt.is_empty() {
                system_parts.push(base_prompt);
            }
            if !tools_section.is_empty() {
                system_parts.push(tools_section);
            }
            let full_system_prompt = system_parts.join("\n\n");

            // Build messages
            let messages = self.conversation.messages();

            // Create completion request
            let request = CompletionRequest {
                model: String::new(),
                messages,
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                top_p: self.config.top_p,
                system_prompt: full_system_prompt,
                api_key_override: String::new(),
            };

            // Call the LLM
            let response = self.provider.complete(&request)?;
            let content = response.content;

            // Check for tool calls
            if tools::has_tool_calls(&content) {
                let calls = tools::parse_tool_calls(&content);
                let _text_part = tools::extract_text(&content);

                // Add assistant message (with tool calls)
                self.conversation.add_assistant_message(&content);

                // Execute each tool call
                for call in &calls {
                    let result = self.tools.execute(call);
                    tool_executions.push(AgentToolExecution {
                        tool_name: call.name.clone(),
                        arguments: call.arguments.clone(),
                        output: result.output.clone(),
                        is_error: result.is_error,
                    });

                    // Add tool result as a system message for the next round
                    let tool_msg = format!(
                        "Tool '{}' returned:\n{}",
                        result.tool_name, result.output
                    );
                    self.conversation.add_message("system", &tool_msg);
                }

                // Continue the loop — the model needs to process tool results
                continue;
            }

            // No tool calls — this is the final response
            self.conversation.add_assistant_message(&content);
            return Ok(AgentResponse {
                text: content,
                tool_calls: tool_executions,
                rounds,
            });
        }
    }

    /// Stream a response token by token, with tool call handling.
    ///
    /// Returns an iterator that yields text tokens. Tool calls are handled
    /// internally — the iterator may pause while tools execute, then resume
    /// with the model's continued response.
    ///
    /// For simplicity, streaming collects the full response first (to parse
    /// tool calls), then yields tokens. True streaming with interleaved tool
    /// execution requires a more complex architecture.
    pub fn run_streaming(
        &mut self,
        user_input: &str,
    ) -> Result<AgentResponse, CrabInferError> {
        // For now, streaming falls back to the non-streaming path.
        // True streaming with tool interleaving is a future enhancement.
        self.run(user_input)
    }

    /// Get knowledge context for the current query.
    fn get_knowledge_context(&self, query: &str) -> Vec<String> {
        match &self.knowledge {
            Some(kb) => kb
                .query_for_prompt(query, self.config.rag_top_k)
                .unwrap_or_default(),
            None => Vec::new(),
        }
    }

    /// Clear conversation history.
    pub fn clear_conversation(&mut self) {
        self.conversation.clear();
    }

    /// Save all persistent state (conversation + facts).
    pub fn save(&self) -> Result<(), CrabInferError> {
        self.conversation.save()?;
        self.facts.save()?;
        if let Some(ref kb) = self.knowledge {
            kb.save()?;
        }
        Ok(())
    }

    /// Get a reference to the facts store (for reading without &mut self).
    pub fn facts_mut_ref(&self) -> &MemoryStore {
        &self.facts
    }

    /// Replace the system prompt.
    pub fn set_system_prompt(&mut self, prompt: SystemPrompt) {
        self.system_prompt = prompt;
    }

    /// Set the conversation persistence path.
    pub fn set_conversation_path(&mut self, path: &str) {
        self.conversation = std::mem::take(&mut self.conversation)
            .with_persist_path(path);
    }

    /// Set the facts persistence path.
    pub fn set_facts_path(&mut self, path: &str) {
        self.facts = std::mem::take(&mut self.facts)
            .with_persist_path(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::parse_tool_calls;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.max_tool_rounds, 10);
        assert_eq!(config.max_tokens, 2048);
    }

    #[test]
    fn test_parse_tool_calls_integration() {
        let response = r#"Let me read that file.
<tool_call>{"name": "file_read", "arguments": {"path": "/etc/hostname"}}</tool_call>"#;

        let calls = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "file_read");
        assert_eq!(calls[0].arguments["path"], "/etc/hostname");
    }

    #[test]
    fn test_tool_registry_prompt_generation() {
        let registry = tools::default_tools();
        let prompt = registry.tools_prompt();
        assert!(prompt.contains("file_read"));
        assert!(prompt.contains("shell_exec"));
        assert!(prompt.contains("<tool_call>"));
    }
}
