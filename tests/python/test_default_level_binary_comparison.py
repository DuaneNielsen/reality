"""
Test that compares C++ generated default_level.lvl with Python generated version.
Simply compares the binary files to ensure they are identical.
"""

import tempfile
from pathlib import Path

from madrona_escape_room.default_level import create_default_level
from madrona_escape_room.level_io import save_compiled_levels


def test_cpp_vs_python_default_level_binary_identical():
    """Test that C++ and Python generate identical binary default level files.

    IMPORTANT: src/default_level.cpp is the ground truth implementation.
    The Python version in madrona_escape_room/default_level.py should be updated
    to match the C++ version exactly if this test fails.
    """
    # Path to C++ generated default level
    cpp_level_path = Path(__file__).parent.parent.parent / "build" / "default_level.lvl"
    assert cpp_level_path.exists(), f"C++ default level file not found at: {cpp_level_path}"

    # Generate Python default level to temporary file
    python_level = create_default_level()

    with tempfile.NamedTemporaryFile(suffix=".lvl", delete=False) as tmp:
        python_level_path = Path(tmp.name)

    try:
        # Save Python level using new unified format (wrapping single level in list)
        save_compiled_levels([python_level], python_level_path)

        # Read both files as binary
        cpp_data = cpp_level_path.read_bytes()
        python_data = python_level_path.read_bytes()

        # Compare binary data
        assert cpp_data == python_data, (
            f"Binary files differ: C++ size={len(cpp_data)}, Python size={len(python_data)}\n"
            f"GROUND TRUTH: src/default_level.cpp\n"
            f"UPDATE REQUIRED: madrona_escape_room/default_level.py to match C++ implementation"
        )

    finally:
        python_level_path.unlink(missing_ok=True)
