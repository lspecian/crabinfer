#!/usr/bin/env python3
"""
CrabInfer Python Client — Minimal Example

Talks to a running `crabinfer serve` instance via the OpenAI-compatible API.

Usage:
    # Terminal 1: Start the server
    crabinfer serve --model /path/to/model.gguf --port 8080

    # Terminal 2: Run this script
    pip install openai
    python client.py
"""

from openai import OpenAI

# Connect to the local CrabInfer server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)


def chat(prompt: str) -> str:
    """Send a single message and return the response."""
    response = client.chat.completions.create(
        model="local",
        messages=[
            {"role": "system", "content": "You are a helpful, concise assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content


def stream_chat(prompt: str):
    """Stream a response token by token."""
    stream = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or "What is Rust?"

    print(f"> {prompt}\n")

    # Non-streaming
    print("--- Non-streaming ---")
    print(chat(prompt))

    # Streaming
    print("\n--- Streaming ---")
    stream_chat(prompt)
