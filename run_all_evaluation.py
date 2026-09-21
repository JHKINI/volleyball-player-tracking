from pathlib import Path
import runpy


SCRIPT = Path(__file__).parent / "src" / "tracking" / "run_all_evaluation.py"

runpy.run_path(str(SCRIPT), run_name="__main__")