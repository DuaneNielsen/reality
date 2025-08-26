"""
One-line patch to make ALL ctypes structures debugger-friendly.
Import this ONCE at the start of your debug session.
"""

import ctypes

# Save original __repr__
_original_structure_repr = ctypes.Structure.__repr__

def _debug_repr(self):
    """Auto-convert arrays to lists in repr for ALL ctypes.Structure instances"""
    parts = []
    if hasattr(self, '_fields_'):
        for field_name, field_type in self._fields_:
            if field_name.startswith('_pad'):
                continue
            
            value = getattr(self, field_name)
            
            # Convert ctypes arrays to lists
            if hasattr(value, '__len__') and hasattr(value, '_type_'):
                value = list(value)
                if len(value) > 20:
                    value = value[:10] + ['...'] + value[-5:]
            
            # Decode char arrays
            elif isinstance(value, bytes):
                value = value.decode('utf-8', errors='ignore').rstrip('\0')
            
            parts.append(f"{field_name}={value!r}")
    
    if len(parts) > 15:
        parts = parts[:15] + ['...']
    
    return f"{self.__class__.__name__}({', '.join(parts)})"

# Monkey-patch ALL ctypes.Structure classes
ctypes.Structure.__repr__ = _debug_repr

print("✅ ctypes.Structure patched - arrays will now display as lists in debugger!")

# To restore original behavior:
def unpatch():
    ctypes.Structure.__repr__ = _original_structure_repr
    print("Restored original ctypes.Structure.__repr__")