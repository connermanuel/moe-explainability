"""Simple path setup for research project.

Add this to the top of any script that needs to import from get_routes:

    import sys
    sys.path.append('.')  # If running from project root
    from setup_paths import *  # This handles all the path setup

    # Now you can import normally
    from scripts.get_routes import extract_ud_routes

Alternative: Just run everything from the project root.
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Convenience imports - now available after importing this module
try:
    from routing import (
        RouterConfig,
        RoutingPipeline,
        cache_to_file,
        extract_ud_routes,
        extract_wordsim_routes,
        get_device_info,
        quick_route_analysis,
    )

    print("✅ Routing extraction tools loaded successfully!")

except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Make sure you're running from the project root directory.")
