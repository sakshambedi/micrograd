import sys
import os
from pathlib import Path

# Add the project root directory to Python path
# This allows imports from the root of the project
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Print useful debugging information when running pytest with -v
def pytest_configure(config):
    if config.option.verbose > 0:
        print(f"Python path: {sys.path}")
        print(f"Project root added to path: {project_root}")