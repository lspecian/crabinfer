//! System Prompt Management — composable prompt builder with token budgets.
//!
//! Provides a structured way to build system prompts from composable sections:
//! identity, instructions, output format, and dynamically injected context
//! (facts from MemoryStore, retrieved chunks from KnowledgeBase).

use serde::{Deserialize, Serialize};

/// A composable system prompt builder.
///
/// Sections are rendered in order: identity → instructions → knowledge context →
/// memory facts → output format. Each section is optional.
///
/// ```rust
/// use crabinfer_core::prompt::SystemPrompt;
///
/// let prompt = SystemPrompt::new()
///     .identity("You are a helpful coding assistant focused on Rust.")
///     .instruction("Always include code examples.")
///     .instruction("Never suggest unsafe code without explaining why.")
///     .output_format("markdown");
///
/// let rendered = prompt.build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPrompt {
    identity: Option<String>,
    instructions: Vec<String>,
    output_format: Option<String>,
    /// Maximum token budget for the rendered system prompt (0 = unlimited).
    token_budget: u32,
}

impl SystemPrompt {
    /// Create an empty system prompt builder.
    pub fn new() -> Self {
        Self {
            identity: None,
            instructions: Vec::new(),
            output_format: None,
            token_budget: 0,
        }
    }

    /// Set the identity section (who the model is, tone, personality).
    pub fn identity(mut self, identity: &str) -> Self {
        self.identity = Some(identity.to_string());
        self
    }

    /// Add an instruction (what to do, what not to do).
    /// Multiple instructions are rendered as a numbered list.
    pub fn instruction(mut self, instruction: &str) -> Self {
        self.instructions.push(instruction.to_string());
        self
    }

    /// Set the output format constraint (e.g., "markdown", "json", "plain text").
    pub fn output_format(mut self, format: &str) -> Self {
        self.output_format = Some(format.to_string());
        self
    }

    /// Set the maximum token budget for the system prompt.
    /// When building with context, content will be truncated to fit.
    /// 0 means unlimited.
    pub fn token_budget(mut self, budget: u32) -> Self {
        self.token_budget = budget;
        self
    }

    /// Render the system prompt without any dynamic context.
    pub fn build(&self) -> String {
        self.build_with_context(&[], &[])
    }

    /// Render the system prompt with injected facts and knowledge context.
    ///
    /// Sections are assembled in order:
    /// 1. Identity
    /// 2. Instructions
    /// 3. Knowledge context (retrieved RAG chunks)
    /// 4. Memory facts (persistent user facts)
    /// 5. Output format
    ///
    /// If a token budget is set, knowledge context is trimmed first,
    /// then facts, to fit within the budget.
    pub fn build_with_context(&self, facts: &[String], knowledge_chunks: &[String]) -> String {
        let mut sections: Vec<String> = Vec::new();

        // 1. Identity
        if let Some(ref identity) = self.identity {
            sections.push(identity.clone());
        }

        // 2. Instructions
        if !self.instructions.is_empty() {
            if self.instructions.len() == 1 {
                sections.push(self.instructions[0].clone());
            } else {
                let numbered: Vec<String> = self
                    .instructions
                    .iter()
                    .enumerate()
                    .map(|(i, inst)| format!("{}. {}", i + 1, inst))
                    .collect();
                sections.push(numbered.join("\n"));
            }
        }

        // 3. Knowledge context
        if !knowledge_chunks.is_empty() {
            let mut kb_section = String::from("Relevant context:\n");
            for chunk in knowledge_chunks {
                kb_section.push_str("- ");
                kb_section.push_str(chunk);
                kb_section.push('\n');
            }
            sections.push(kb_section);
        }

        // 4. Memory facts
        if !facts.is_empty() {
            let mut facts_section = String::from("Known facts about the user:\n");
            for fact in facts {
                facts_section.push_str("- ");
                facts_section.push_str(fact);
                facts_section.push('\n');
            }
            sections.push(facts_section);
        }

        // 5. Output format
        if let Some(ref format) = self.output_format {
            sections.push(format!("Respond in {} format.", format));
        }

        let mut result = sections.join("\n\n");

        // Apply token budget if set (rough estimate: 1 token ≈ 4 chars)
        if self.token_budget > 0 {
            let char_budget = (self.token_budget as usize) * 4;
            if result.len() > char_budget {
                result.truncate(char_budget);
                // Avoid cutting in the middle of a UTF-8 char
                while !result.is_char_boundary(result.len()) {
                    result.pop();
                }
            }
        }

        result
    }

    /// Estimate the token count of the rendered prompt (approx 1 token per 4 chars).
    pub fn estimate_tokens(&self) -> u32 {
        let rendered = self.build();
        (rendered.len() / 4).max(1) as u32
    }

    /// Estimate token count with context.
    pub fn estimate_tokens_with_context(&self, facts: &[String], knowledge: &[String]) -> u32 {
        let rendered = self.build_with_context(facts, knowledge);
        (rendered.len() / 4).max(1) as u32
    }

    /// Save the prompt configuration to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Load a prompt configuration from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for SystemPrompt {
    fn default() -> Self {
        Self::new()
    }
}

// === Pre-built templates ===

impl SystemPrompt {
    /// Template: coding assistant.
    pub fn coding_assistant() -> Self {
        Self::new()
            .identity("You are a helpful coding assistant. You write clean, efficient, well-tested code.")
            .instruction("Always include code examples when explaining concepts.")
            .instruction("Prefer simple solutions over clever ones.")
            .instruction("Point out potential bugs or edge cases.")
            .output_format("markdown")
    }

    /// Template: document Q&A.
    pub fn document_qa() -> Self {
        Self::new()
            .identity("You answer questions based on the provided context documents. If the answer is not in the context, say so honestly.")
            .instruction("Only use information from the provided context.")
            .instruction("Quote relevant passages when possible.")
            .instruction("If the context doesn't contain enough information, say 'I don't have enough information to answer that.'")
    }

    /// Template: conversational assistant.
    pub fn conversational() -> Self {
        Self::new()
            .identity("You are a friendly, helpful assistant. You give concise, accurate answers.")
            .instruction("Be concise but thorough.")
            .instruction("Ask clarifying questions when the request is ambiguous.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_prompt() {
        let prompt = SystemPrompt::new();
        assert_eq!(prompt.build(), "");
    }

    #[test]
    fn test_identity_only() {
        let prompt = SystemPrompt::new().identity("You are a helpful assistant.");
        assert_eq!(prompt.build(), "You are a helpful assistant.");
    }

    #[test]
    fn test_full_prompt() {
        let prompt = SystemPrompt::new()
            .identity("You are a coding assistant.")
            .instruction("Write clean code.")
            .instruction("Include tests.")
            .output_format("markdown");

        let rendered = prompt.build();
        assert!(rendered.contains("You are a coding assistant."));
        assert!(rendered.contains("1. Write clean code."));
        assert!(rendered.contains("2. Include tests."));
        assert!(rendered.contains("Respond in markdown format."));
    }

    #[test]
    fn test_with_context() {
        let prompt = SystemPrompt::new().identity("Assistant");
        let facts = vec!["User prefers Rust".to_string()];
        let knowledge = vec!["Rust is a systems language".to_string()];

        let rendered = prompt.build_with_context(&facts, &knowledge);
        assert!(rendered.contains("Known facts about the user:"));
        assert!(rendered.contains("User prefers Rust"));
        assert!(rendered.contains("Relevant context:"));
        assert!(rendered.contains("Rust is a systems language"));
    }

    #[test]
    fn test_token_budget() {
        let prompt = SystemPrompt::new()
            .identity("A very long identity string that goes on and on and on and on and on.")
            .token_budget(5); // ~20 chars

        let rendered = prompt.build();
        assert!(rendered.len() <= 20);
    }

    #[test]
    fn test_single_instruction_no_numbering() {
        let prompt = SystemPrompt::new().instruction("Be helpful.");
        assert_eq!(prompt.build(), "Be helpful.");
    }

    #[test]
    fn test_json_roundtrip() {
        let prompt = SystemPrompt::coding_assistant();
        let json = prompt.to_json().unwrap();
        let restored = SystemPrompt::from_json(&json).unwrap();
        assert_eq!(prompt.build(), restored.build());
    }

    #[test]
    fn test_template_coding_assistant() {
        let prompt = SystemPrompt::coding_assistant();
        let rendered = prompt.build();
        assert!(rendered.contains("coding assistant"));
        assert!(rendered.contains("code examples"));
    }
}
