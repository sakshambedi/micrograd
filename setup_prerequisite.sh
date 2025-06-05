#!/usr/bin/env bash

set -e

echo "Detecting OS..."
OS="$(uname)"
echo "OS detected: $OS"

if [[ "$OS" == "Darwin" ]]; then
    # macOS
    echo "Installing prerequisites for macOS..."
    # Install Homebrew if not present
    if ! command -v brew &>/dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew update
    brew install cmake eigen python@3 git
elif [[ "$OS" == "Linux" ]]; then
    # Linux (Debian/Ubuntu)
    echo "Installing prerequisites for Linux..."
    sudo apt-get update
    sudo apt-get install -y cmake g++ python3 python3-dev python3-pip git
    # Eigen is available as a package
    sudo apt-get install -y libeigen3-dev
else
    echo "Unsupported OS: $OS"
    exit 1
fi

echo "All prerequisites installed."
echo "You can now build the project with:"
echo "  mkdir -p build && cd build"
echo "  cmake .."
echo "  make"
