# Reindex the source files
Follow the following procedure 

1. list the files in the projects key directories

```bash
cd ~/madrona_escape_room && ls src/*pp && ls madrona_escape_room/*.py && ls codegen/*.py
```

2. For each file read the file and undersand its contents

3. Create an index in the following format, by copying the example below, and write it to FILEINDEX.md in theproject root directory

src\consts.hpp - constants
src\types.hpp - types
src\mgr.cpp - setup, start and step the simulation 
src\sim.cpp - simulation, runs on gpu
codegen\generate_dataclass_structs.py - generates dataclasses in madrona_escape_room\generated_dataclasses.py
madrona_escape_room\__init__.py - imports for module
madrona_escape_room\manager.py - python interface to src\mgr.cpp


