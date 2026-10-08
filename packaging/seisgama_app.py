"""PyInstaller entry point: GUI by default, command line if arguments are given."""
import multiprocessing
import os
import sys


def _attach_console():
    """A --windowed Windows build has no stdout; reuse the console it was started from."""
    if sys.platform == "win32" and len(sys.argv) > 1 and sys.stdout is None:
        import ctypes

        if ctypes.windll.kernel32.AttachConsole(-1):
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace")
        else:
            sys.stdout = sys.stderr = open(os.devnull, "w")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _attach_console()
    from seisgama.cli import main

    sys.exit(main())
