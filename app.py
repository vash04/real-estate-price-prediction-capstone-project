from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
NESTED_APP = ROOT / "dsmp-capstone-project" / "streamlit_app" / "app.py"

if __name__ == "__main__":
    runpy.run_path(str(NESTED_APP), run_name="__main__")
