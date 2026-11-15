from pathlib import Path
import sys

source_root = Path(__file__).resolve().parents[1] / "src"
if source_root.is_dir():
    sys.path.insert(0, str(source_root))

from llm.schema import main


if __name__ == "__main__":
    raise SystemExit(main())
