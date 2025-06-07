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

        # Add Homebrew to PATH for the current session if it was just installed
        if [[ -f "/opt/homebrew/bin/brew" ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [[ -f "/usr/local/bin/brew" ]]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
        echo "Homebrew installed and configured."
    fi

    echo "Updating Homebrew and installing packages..."
    brew update
    brew install cmake eigen python@3 git
elif [[ "$OS" == "Linux" ]]; then
    # Linux (Debian/Ubuntu)
    echo "Installing prerequisites for Linux..."

    # Check if we can use apt-get (Debian/Ubuntu)
    if command -v apt-get &>/dev/null; then
        echo "Detected Debian/Ubuntu-based distribution"
        sudo apt-get update
        sudo apt-get install -y cmake g++ python3 python3-dev python3-pip git libeigen3-dev
    # Check if we can use dnf (Fedora/RHEL)
    elif command -v dnf &>/dev/null; then
        echo "Detected Fedora/RHEL-based distribution"
        sudo dnf update -y
        sudo dnf install -y cmake gcc-c++ python3 python3-devel python3-pip git eigen3-devel
    # Check if we can use pacman (Arch Linux)
    elif command -v pacman &>/dev/null; then
        echo "Detected Arch-based distribution"
        sudo pacman -Syu --noconfirm
        sudo pacman -S --noconfirm cmake gcc python python-pip git eigen
    else
        echo "Unsupported Linux distribution. Please install the following packages manually:"
        echo "- cmake"
        echo "- C++ compiler (g++ or clang++)"
        echo "- Python 3 with development headers"
        echo "- pip for Python 3"
        echo "- git"
        echo "- Eigen 3 library"
        exit 1
    fi
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Verify installations
echo "Verifying installations..."

if ! command -v cmake &>/dev/null; then
    echo "WARNING: cmake not found in PATH. Installation may have failed."
else
    CMAKE_VERSION=$(cmake --version | head -n 1)
    echo "✓ $CMAKE_VERSION"
fi

if ! command -v python3 &>/dev/null; then
    echo "WARNING: python3 not found in PATH. Installation may have failed."
else
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION"
fi

if ! command -v git &>/dev/null; then
    echo "WARNING: git not found in PATH. Installation may have failed."
else
    GIT_VERSION=$(git --version)
    echo "✓ $GIT_VERSION"
fi

echo ""
echo "All prerequisites installed."
echo "You can now build the project with:"
echo "  ./run.sh"
echo "Or manually with:"
echo "  mkdir -p build && cd build"
echo "  cmake .."
echo "  make"
