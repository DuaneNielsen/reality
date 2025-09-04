# The default ``config.py``
# flake8: noqa


def set_prefs(prefs):
    """This function is called before opening the project"""

    # Specify which files and folders to ignore in the project.
    # Changes to ignored resources are not added to the history and
    # VCSs.  Also they are not returned in `Project.get_files()`.
    # Note that ``?`` and ``*`` match all characters but slashes.
    # '*.pyc': matches 'test.pyc' and 'pkg/test.pyc'
    # 'mod*.py': matches 'test/mod1.py' but not 'mod/1.py'
    # '.svn': matches 'pkg/.svn' and all of its children
    # 'build/*.o': matches 'build/lib.o' but not 'build/sub/lib.o'
    # 'build//*.o': matches 'build/lib.o' and 'build/sub/lib.o'
    prefs["ignored_resources"] = [
        "*.pyc",
        "*~",
        ".ropeproject",
        ".hg",
        ".svn",
        "_svn",
        ".git",
        ".tox",
        "external*",
        "scratch*",
        "build*",
        ".venv*",
    ]

    # Specifies which files should be considered python files.  It is
    # useful when you have scripts inside your project.  Only files
    # ending with ``.py`` are considered to be python files by
    # default.
    # prefs['python_files'] = ['*.py']

    # Custom source folders:  By default rope searches the project
    # for finding source folders (folders that should be searched
    # for finding modules).  You can add paths to that list.  Note
    # that rope guesses project source folders correctly most of the
    # time; use this if you have any problems.
    # The folders should be relative to project root and use '/' for
    # separating folders regardless of the platform rope is running on.
    # 'src/my_source_folder' for instance.
    # prefs['source_folders'] = []

    # You can extend the list of builtin and c-extension modules by
    # importing and extending this list.
    # from rope.base.oi.type_hinting.py3_type_hinting import builtin_libs
    # prefs['builtin_libs'] = builtin_libs[:]

    # Performance settings (the default is 3 seconds):
    # prefs['disk_timeout'] = 3

    # Save history across sessions:
    # prefs['save_history'] = True
    # prefs['compress_history'] = False
    # prefs['max_history_items'] = 32


def project_opened(project):
    """This function is called whenever a project is opened"""
    pass
