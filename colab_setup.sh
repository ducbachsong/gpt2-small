#!/usr/bin/env bash
# Colab (or any Linux box): .NET 8 into ~/.dotnet, and the Python packages.
#     !bash colab_setup.sh && python traingpt2cs.py
set -e
if ! command -v dotnet >/dev/null && [ ! -x "$HOME/.dotnet/dotnet" ]; then
    curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 8.0 --install-dir "$HOME/.dotnet"
fi
"$HOME/.dotnet/dotnet" --version 2>/dev/null || dotnet --version
pip install -q -r requirements.txt
