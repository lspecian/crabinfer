//! `crabinfer mcp` — manage MCP servers and run the built-in MCP server.

use crabinfer_core::mcp::{McpServer, McpServerConfig, McpServerRegistry, McpTransport};

/// Default config path.
fn config_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    format!("{}/.crabinfer/mcp-servers.json", home)
}

/// List configured MCP servers.
pub fn list() {
    let registry = McpServerRegistry::load(&config_path());

    if registry.count() == 0 {
        eprintln!("No MCP servers configured.");
        eprintln!();
        eprintln!("Add one with:");
        eprintln!("  crabinfer mcp add --name <name> --transport stdio --command <cmd> [--args <arg1> <arg2> ...]");
        eprintln!("  crabinfer mcp add --name <name> --transport http --command <url>");
        return;
    }

    println!("{:<20} {:<8} {:<10} {}", "NAME", "TYPE", "STATUS", "COMMAND");
    println!("{}", "-".repeat(70));

    for server in registry.servers() {
        let transport = match server.transport {
            McpTransport::Stdio => "stdio",
            McpTransport::Http => "http",
        };
        let status = if server.enabled { "enabled" } else { "disabled" };
        let cmd = if server.args.is_empty() {
            server.command.clone()
        } else {
            format!("{} {}", server.command, server.args.join(" "))
        };
        println!("{:<20} {:<8} {:<10} {}", server.name, transport, status, cmd);
    }
}

/// Add a new MCP server.
pub fn add(name: &str, transport: &str, command: &str, args: &[String], description: &str) {
    let transport = match transport {
        "stdio" => McpTransport::Stdio,
        "http" => McpTransport::Http,
        _ => {
            eprintln!("Unknown transport: {transport}. Use 'stdio' or 'http'.");
            std::process::exit(1);
        }
    };

    let config = McpServerConfig {
        name: name.to_string(),
        transport,
        command: command.to_string(),
        args: args.to_vec(),
        enabled: true,
        description: description.to_string(),
    };

    let path = config_path();
    let mut registry = McpServerRegistry::load(&path).with_persist_path(&path);
    registry.add(config);

    match registry.save() {
        Ok(()) => println!("Added MCP server '{name}'."),
        Err(e) => {
            eprintln!("Failed to save: {e}");
            std::process::exit(1);
        }
    }
}

/// Remove an MCP server.
pub fn remove(name: &str) {
    let path = config_path();
    let mut registry = McpServerRegistry::load(&path).with_persist_path(&path);

    if registry.remove(name) {
        match registry.save() {
            Ok(()) => println!("Removed MCP server '{name}'."),
            Err(e) => eprintln!("Failed to save: {e}"),
        }
    } else {
        eprintln!("No server named '{name}'.");
    }
}

/// Enable or disable an MCP server.
pub fn toggle(name: &str, enable: bool) {
    let path = config_path();
    let mut registry = McpServerRegistry::load(&path).with_persist_path(&path);

    if registry.set_enabled(name, enable) {
        match registry.save() {
            Ok(()) => {
                let state = if enable { "enabled" } else { "disabled" };
                println!("Server '{name}' {state}.");
            }
            Err(e) => eprintln!("Failed to save: {e}"),
        }
    } else {
        eprintln!("No server named '{name}'.");
    }
}

/// Test connecting to an MCP server and list its tools.
pub fn test_server(name: &str) {
    let registry = McpServerRegistry::load(&config_path());

    let config = match registry.get(name) {
        Some(c) => c,
        None => {
            eprintln!("No server named '{name}'.");
            std::process::exit(1);
        }
    };

    eprintln!("Connecting to '{}'...", config.name);

    match crabinfer_core::mcp::connect_server(config) {
        Ok(client) => {
            match client.list_tools() {
                Ok(tools) => {
                    println!("Connected to '{}' — {} tools available:", name, tools.len());
                    for tool in &tools {
                        println!("  {} — {}", tool.name, tool.description);
                    }
                }
                Err(e) => {
                    eprintln!("Connected but failed to list tools: {e}");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to connect: {e}");
            std::process::exit(1);
        }
    }
}

/// Run CrabInfer as an MCP server (stdio transport).
///
/// This allows external tools (Claude Desktop, other agents) to use
/// CrabInfer's built-in tools via the MCP protocol.
pub fn serve() {
    eprintln!("Starting CrabInfer MCP server (stdio)...");
    eprintln!("Tools: file_read, file_write, file_list, shell_exec, web_fetch");
    eprintln!("Listening on stdin/stdout (JSON-RPC 2.0)");

    let server = McpServer::new();
    if let Err(e) = server.run() {
        eprintln!("MCP server error: {e}");
        std::process::exit(1);
    }
}
