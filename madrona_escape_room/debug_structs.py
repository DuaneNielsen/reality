"""
Debug helpers for ctypes structures - standard approach used by most projects
"""

import ctypes
from typing import Any


def debug(obj: Any) -> dict:
    """
    Convert ctypes Structure to debugger-friendly dict.
    This is the standard approach used by Apache TVM, pyclibrary, etc.
    
    Usage in debugger:
        from madrona_escape_room.debug_structs import debug
        debug(level)  # Shows all fields with arrays as lists
    """
    if not isinstance(obj, ctypes.Structure):
        return obj
        
    result = {}
    for name, ctype in obj._fields_:
        if name.startswith('_pad'):
            continue
            
        value = getattr(obj, name)
        
        # Convert arrays to lists
        if hasattr(value, '__len__') and hasattr(value, '_type_'):
            # Truncate large arrays
            arr = list(value)
            if len(arr) > 20:
                value = arr[:10] + ['...'] + arr[-5:]
            else:
                value = arr
        # Decode char arrays
        elif isinstance(value, (bytes, ctypes.Array)):
            try:
                if hasattr(value, '_type_') and value._type_ == ctypes.c_char:
                    value = value.value.decode('utf-8', errors='ignore').rstrip('\0')
            except:
                pass
                
        result[name] = value
    
    return result


def pp(obj: Any) -> None:
    """Pretty print a ctypes Structure"""
    import pprint
    pprint.pprint(debug(obj))


# For convenience in debugger, also provide a base class approach
# This is what Apache TVM and others do
class DebugStructure(ctypes.Structure):
    """Base class that auto-formats arrays for debugging"""
    
    def __repr__(self):
        parts = []
        for name, _ in self._fields_:
            if name.startswith('_pad'):
                continue
            value = getattr(self, name)
            
            # Convert arrays to lists for display
            if hasattr(value, '__len__') and hasattr(value, '_type_'):
                if len(value) > 10:
                    value = f"[{list(value)[:3]}...{len(value)} items]"
                else:
                    value = list(value)
            elif isinstance(value, bytes):
                value = value.decode('utf-8', errors='ignore').rstrip('\0')
                
            parts.append(f"{name}={value!r}")
            
        # Limit output for readability
        if len(parts) > 10:
            parts = parts[:10] + ['...']
            
        return f"{self.__class__.__name__}({', '.join(parts)})"
    
    def _asdict(self):
        """Convert to dict - common pattern in ctypes projects"""
        return debug(self)