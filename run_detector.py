"""Source-tree entry point; installation is not required."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chrompeak.detector import main


if __name__ == "__main__":
    raise SystemExit(main())
