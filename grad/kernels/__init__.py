# This file marks the grad.kernels directory as a Python package

# Try to import the compiled cpu_kernel extension
try:
    from grad.kernels import cpu_kernel
except ImportError:
    # If import fails, provide a helpful message
    import sys
    print("Warning: Could not import cpu_kernel extension. Make sure it's properly compiled.", file=sys.stderr)