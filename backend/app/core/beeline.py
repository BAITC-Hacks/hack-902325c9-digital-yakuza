import hashlib
import sys
from pathlib import Path

from app.core.config import get_settings

RUN_TIMEOUT_SECONDS = 600
RUNTIME_FILES = (
    "agent.py", "environment.py", "mock_environment.py", "scoring_core.py",
    "local_eval.py", "make_submission.py", "workflow.py",
    "customer_profile.csv", "tariff_dictionary.csv", "feature_dictionary.csv",
    "data/change_tariff.csv", "data/dict_tariff.csv",
)


def agent_directory() -> Path:
    root = Path(__file__).resolve().parents[3]
    configured = get_settings().beeline_agent_dir
    directory = (root / (configured if configured is not None else "beeline_agent")).resolve()
    missing = [name for name in RUNTIME_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Beeline agent directory {directory} is missing: {', '.join(missing)}. "
            "Set BEELINE_AGENT_DIR or provide beeline_agent at the repository root."
        )
    for name in RUNTIME_FILES:
        if not name.endswith(".py"):
            continue
        module = sys.modules.get(Path(name).stem)
        if module is not None:
            location = getattr(module, "__file__", None)
            if location is None or Path(location).resolve() != directory / name:
                raise ValueError(f"Conflicting Beeline module {name}; restart with the correct BEELINE_AGENT_DIR")
    path = str(directory)
    if not sys.path or sys.path[0] != path:
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    return directory


def agent_sha256() -> str:
    return hashlib.sha256((agent_directory() / "agent.py").read_bytes()).hexdigest()
