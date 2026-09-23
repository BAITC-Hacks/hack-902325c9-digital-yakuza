import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

BACKEND = Path(__file__).resolve().parents[1]
SOURCE = BACKEND.parent / "beeline_agent"
FILES = (
    "agent.py", "environment.py", "mock_environment.py", "scoring_core.py",
    "local_eval.py", "make_submission.py", "requirements.txt", "workflow.py",
    "customer_profile.csv", "tariff_dictionary.csv", "feature_dictionary.csv",
    "data/change_tariff.csv", "data/dict_tariff.csv",
)


def main():
    contents = {name: (SOURCE / name).read_bytes() for name in FILES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    contents["manifest.json"] = json.dumps(hashes, sort_keys=True, indent=2).encode()
    destination = BACKEND / "vendor" / "beeline_agent.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
    print(f"Packaged {len(FILES)} unchanged files: {destination}")
    print(f"Agent SHA256: {hashes['agent.py']}")


if __name__ == "__main__":
    main()
