import sys
from pathlib import Path

# Ensure the 'src' directory is in the Python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from generator import main

if __name__ == "__main__":
    main()
