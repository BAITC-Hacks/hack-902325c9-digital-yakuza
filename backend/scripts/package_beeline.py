import argparse
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

BACKEND = Path(__file__).resolve().parents[1]
SOURCE = BACKEND.parent / "beeline_agent"
DESTINATION = BACKEND / "vendor" / "beeline_agent.zip"
FILES = (
    "agent.py", "environment.py", "mock_environment.py", "scoring_core.py",
    "local_eval.py", "make_submission.py", "requirements.txt",
    "customer_profile.csv", "tariff_dictionary.csv", "feature_dictionary.csv",
    "data/change_tariff.csv", "data/dict_tariff.csv",
)


def check_package(source=SOURCE, destination=DESTINATION):
    try:
        hashes = {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in FILES
        }
        with ZipFile(destination) as archive:
            names = archive.namelist()
            if len(names) != len(FILES) + 1 or set(names) != set(FILES) | {"manifest.json"}:
                return False
            manifest = json.loads(archive.read("manifest.json"))
            return manifest == hashes and all(
                hashlib.sha256(archive.read(name)).hexdigest() == hashes[name]
                for name in FILES
            )
    except (OSError, BadZipFile, KeyError, ValueError, RuntimeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        current = check_package()
        print(f"AGENT PACKAGE: {'CURRENT' if current else 'OUTDATED'}")
        return 0 if current else 1
    contents = {name: (SOURCE / name).read_bytes() for name in FILES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    contents["manifest.json"] = json.dumps(hashes, sort_keys=True, indent=2).encode()
    destination = DESTINATION
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
    print(f"Packaged {len(FILES)} unchanged files: {destination}")
    print(f"Agent SHA256: {hashes['agent.py']}")
    if not check_package():
        raise RuntimeError("Agent package verification failed")
    print("AGENT PACKAGE: CURRENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
