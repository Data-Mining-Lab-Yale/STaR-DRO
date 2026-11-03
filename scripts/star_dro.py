"""Source-checkout command-line entry point for STaR-DRO."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from star_dro.cli import main
if __name__ == "__main__":
    main()
