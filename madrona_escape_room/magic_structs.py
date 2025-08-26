"""
Make ctypes arrays behave like Python lists automatically.
This uses __getattribute__ to intercept access and return actual lists.
"""

import ctypes
from typing import Any


class MagicStructure(ctypes.Structure):
    """
    A ctypes.Structure where arrays are automatically returned as lists.
    
    When you access myobj.spawn_x, you get a Python list, not a ctypes array.
    This makes debugging trivial - the debugger sees actual lists!
    """
    
    def __getattribute__(self, name: str) -> Any:
        """Intercept attribute access to auto-convert arrays"""
        # Get the raw value first
        value = object.__getattribute__(self, name)
        
        # If it's a ctypes array, return it as a list
        if hasattr(value, '__len__') and hasattr(value, '_type_'):
            return list(value)
        
        # If it's a char array (bytes), decode to string
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore').rstrip('\0')
            
        return value
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting arrays from lists"""
        # Check if this field exists and is an array
        if hasattr(self.__class__, '_fields_'):
            for field_name, field_type in self._fields_:
                if field_name == name and hasattr(field_type, '_length_'):
                    # It's an array field - get the raw array
                    raw_array = object.__getattribute__(self, name)
                    # Copy values from list/tuple to ctypes array
                    if isinstance(value, (list, tuple)):
                        for i, v in enumerate(value[:len(raw_array)]):
                            raw_array[i] = v
                        return
        
        # Normal attribute setting
        object.__setattr__(self, name, value)


def magicify_struct(struct_class: type) -> type:
    """
    Transform an existing ctypes.Structure class to use magic properties.
    This creates a new class that wraps the original.
    
    Usage:
        from madrona_escape_room.generated_structs import CompiledLevel
        MagicCompiledLevel = magicify_struct(CompiledLevel)
        
        # Now use MagicCompiledLevel instead
        level = MagicCompiledLevel()
        level.spawn_x  # Returns a Python list, not ctypes array!
    """
    
    class MagicWrapper(struct_class):
        def __getattribute__(self, name: str) -> Any:
            value = super().__getattribute__(name)
            
            # Auto-convert arrays to lists
            if hasattr(value, '__len__') and hasattr(value, '_type_'):
                return list(value)
            
            # Decode char arrays
            if isinstance(value, bytes):
                return value.decode('utf-8', errors='ignore').rstrip('\0')
                
            return value
        
        def __setattr__(self, name: str, value: Any) -> None:
            # Check if this is an array field
            for field_name, field_type in self._fields_:
                if field_name == name and hasattr(field_type, '_length_'):
                    # Get the raw array and update it
                    raw_array = super().__getattribute__(name)
                    if isinstance(value, (list, tuple)):
                        for i, v in enumerate(value[:len(raw_array)]):
                            raw_array[i] = v
                        return
            
            super().__setattr__(name, value)
        
        def get_raw(self, name: str) -> Any:
            """Get the raw ctypes array if you need it"""
            return super().__getattribute__(name)
    
    MagicWrapper.__name__ = f'Magic{struct_class.__name__}'
    return MagicWrapper


# Alternative: Use a wrapper that doesn't inherit from Structure
class StructWrapper:
    """
    Wraps a ctypes.Structure to make it fully Pythonic.
    Arrays become lists, you can use dict access, etc.
    """
    
    def __init__(self, struct_instance):
        self._struct = struct_instance
    
    def __getattr__(self, name: str) -> Any:
        value = getattr(self._struct, name)
        
        # Convert arrays to lists
        if hasattr(value, '__len__') and hasattr(value, '_type_'):
            return list(value)
        
        # Decode strings
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore').rstrip('\0')
            
        return value
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name == '_struct':
            object.__setattr__(self, name, value)
            return
        
        # Try to set on the wrapped struct
        attr = getattr(self._struct, name)
        if hasattr(attr, '__len__') and hasattr(attr, '_type_'):
            # It's an array - copy values
            for i, v in enumerate(value[:len(attr)]):
                attr[i] = v
        else:
            setattr(self._struct, name, value)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like setting"""
        setattr(self, key, value)
    
    def to_dict(self) -> dict:
        """Convert to a plain Python dict"""
        result = {}
        for field_name, _ in self._struct._fields_:
            if not field_name.startswith('_pad'):
                result[field_name] = getattr(self, field_name)
        return result
    
    def __repr__(self) -> str:
        return f"StructWrapper({self.to_dict()})"
    
    @property
    def raw(self):
        """Access the underlying ctypes structure"""
        return self._struct


# Usage examples:
"""
# Option 1: Replace your import
from madrona_escape_room.magic_structs import magicify_struct
from madrona_escape_room.generated_structs import CompiledLevel

CompiledLevel = magicify_struct(CompiledLevel)  # Transform it
level = CompiledLevel()
print(level.spawn_x)  # It's a list!

# Option 2: Use wrapper
from madrona_escape_room.generated_structs import CompiledLevel
from madrona_escape_room.magic_structs import StructWrapper

level = CompiledLevel()
wrapped = StructWrapper(level)
print(wrapped.spawn_x)  # List!
print(wrapped['spawn_x'])  # Dict-like access!
print(wrapped.to_dict())  # Full dict

# Option 3: Inherit from MagicStructure when defining
from madrona_escape_room.magic_structs import MagicStructure

class MyStruct(MagicStructure):
    _fields_ = [
        ("array_field", ctypes.c_float * 10),
    ]

s = MyStruct()
print(s.array_field)  # It's a list!
"""