from pathlib import Path
import runpy


SCRIPT = Path(__file__).parent / "src" / "tracking" / "fix_numpy_compat.py"

runpy.run_path(str(SCRIPT), run_name="__main__")