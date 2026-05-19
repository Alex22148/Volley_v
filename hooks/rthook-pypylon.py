# hooks\rthook-pypylon.py
import os
pylon_candidates = [
    r"C:\Program Files\Basler\pylon 8\Runtime\x64",
    r"C:\Program Files\Basler\pylon 7\Runtime\x64",
]
if hasattr(os, "add_dll_directory"):
    for p in pylon_candidates:
        if os.path.isdir(p):
            os.add_dll_directory(p)
