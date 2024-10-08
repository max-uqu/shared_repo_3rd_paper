from pathlib import Path


def make_dir_if_not_exists(path: Path):
    directory = Path(path)

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {directory}")
