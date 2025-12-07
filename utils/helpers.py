"""
Utility helper functions
"""
import os
import json
from typing import Any, Dict


def ensure_directories(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file and return as dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return {}


def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """Save dictionary to JSON file."""
    try:
        ensure_directories(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def get_color_gradient(value: float, low_color: str = "#10B981", high_color: str = "#EF4444") -> str:
    """
    Get color between low and high based on value (0-1).
    Returns hex color string.
    """
    # Parse hex colors
    low_r = int(low_color[1:3], 16)
    low_g = int(low_color[3:5], 16)
    low_b = int(low_color[5:7], 16)
    
    high_r = int(high_color[1:3], 16)
    high_g = int(high_color[3:5], 16)
    high_b = int(high_color[5:7], 16)
    
    # Interpolate
    r = int(low_r + (high_r - low_r) * value)
    g = int(low_g + (high_g - low_g) * value)
    b = int(low_b + (high_b - low_b) * value)
    
    return f"#{r:02x}{g:02x}{b:02x}"
