class Crabinfer < Formula
  desc "Safe, memory-aware LLM inference engine for Apple Silicon"
  homepage "https://github.com/lspecian/crabinfer"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/lspecian/crabinfer/releases/download/v#{version}/crabinfer-darwin-arm64.tar.gz"
      # sha256 "PLACEHOLDER" # Updated by release CI
    else
      url "https://github.com/lspecian/crabinfer/releases/download/v#{version}/crabinfer-darwin-x64.tar.gz"
      # sha256 "PLACEHOLDER" # Updated by release CI
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/lspecian/crabinfer/releases/download/v#{version}/crabinfer-linux-arm64.tar.gz"
      # sha256 "PLACEHOLDER" # Updated by release CI
    else
      url "https://github.com/lspecian/crabinfer/releases/download/v#{version}/crabinfer-linux-x64.tar.gz"
      # sha256 "PLACEHOLDER" # Updated by release CI
    end
  end

  def install
    bin.install "crabinfer"
    bin.install "crabinfer-server"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/crabinfer --version")
  end
end
