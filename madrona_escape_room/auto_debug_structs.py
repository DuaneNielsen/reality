"""
Automatic debugger-friendly ctypes structures using __getattr__
This is the cleanest solution - works automatically in debuggers!
"""

import ctypes
from typing import Any


class AutoDebugStructure(ctypes.Structure):
    """
    ctypes.Structure that automatically converts arrays to lists when accessed.
    This makes debugging MUCH easier - arrays display as lists in debuggers!

    Based on the pattern used by major projects but with automatic conversion.
    """

    def __getattribute__(self, name: str) -> Any:
        """Override to auto-convert arrays to lists when accessed"""
        value = super().__getattribute__(name)

        # If it's a ctypes array, convert to list automatically
        if hasattr(value, "__len__") and hasattr(value, "_type_"):
            return list(value)

        # If it's a char array (bytes), decode to string
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").rstrip("\0")

        return value

    def __repr__(self) -> str:
        """Clean representation for debugging"""
        parts = []
        for field_name, field_type in self._fields_:
            if field_name.startswith("_pad"):
                continue

            # Use getattr which triggers our __getattribute__ conversion
            value = getattr(self, field_name)

            # Truncate large lists for readability
            if isinstance(value, list) and len(value) > 10:
                value = value[:5] + ["..."] + value[-2:]

            parts.append(f"{field_name}={value!r}")

        # Limit fields shown for readability
        if len(parts) > 10:
            parts = parts[:10] + ["..."]

        return f"{self.__class__.__name__}({', '.join(parts)})"


# Alternative: Monkey-patch existing structures with properties
def add_debug_properties(cls: type) -> type:
    """
    Add properties to existing ctypes.Structure for automatic array conversion.
    Use this on generated structures you can't modify.
    """
    for field_name, field_type in cls._fields_:
        if field_name.startswith("_pad"):
            continue

        # Check if it's an array type
        if hasattr(field_type, "_length_"):
            # Create a property that returns list
            def make_property(name):
                private_name = f"_{name}_raw"

                def getter(self):
                    # Get raw array
                    if not hasattr(self, private_name):
                        setattr(self, private_name, super(cls, self).__getattribute__(name))
                    raw = getattr(self, private_name)
                    return list(raw)

                def setter(self, value):
                    # Get raw array and update it
                    if not hasattr(self, private_name):
                        setattr(self, private_name, super(cls, self).__getattribute__(name))
                    raw = getattr(self, private_name)
                    for i, v in enumerate(value):
                        raw[i] = v

                return property(getter, setter)

            setattr(cls, field_name, make_property(field_name))

    # Add better __repr__
    original_repr = cls.__repr__

    def new_repr(self):
        parts = []
        for field_name, _ in self._fields_:
            if field_name.startswith("_pad"):
                continue
            value = getattr(self, field_name)
            if isinstance(value, list) and len(value) > 10:
                value = value[:5] + ["..."] + value[-2:]
            parts.append(f"{field_name}={value!r}")
        if len(parts) > 10:
            parts = parts[:10] + ["..."]
        return f"{self.__class__.__name__}({', '.join(parts)})"

    cls.__repr__ = new_repr
    return cls
