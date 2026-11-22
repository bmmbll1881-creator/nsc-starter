"""Compare all course architecture fields with the pinned upstream YAML records."""

from io import BytesIO
import json
from zipfile import ZipFile

from prepare_data import DATA, download


REVISION = "47770415327900684ed95c0bc9bf7cd0d0047fad"
ARCHIVE = f"https://codeload.github.com/microsoft/archai/zip/{REVISION}"
PANEL = f"archai-{REVISION}/archai/nlp/nas/saved_logs/random_GPT2_wt103"


def scalar(value):
    """Decode the scalar types in the archived flat YAML schema."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def decode_config(contents):
    """Decode archived scalar fields and block sequences without a YAML dependency."""
    config = {}
    for line in contents.decode("utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("- "):
            config[key].append(scalar(line[2:]))
        else:
            key, value = line.split(":", 1)
            config[key] = scalar(value.strip()) if value.strip() else []
    return config


def main():
    """Stop with a nonzero exit if any upstream configuration differs."""
    local = json.loads((DATA / "architectures.json").read_text(encoding="utf-8"))
    mismatches = []
    with ZipFile(BytesIO(download(ARCHIVE))) as archive:
        for name, config in local.items():
            config_dir, trial_dir = name.rsplit("_", 1)
            candidate = archive.read(f"{PANEL}/{config_dir}/{trial_dir}/model_config.yaml")
            upstream = decode_config(candidate)
            changed = sorted(
                field for field in config.keys() | upstream.keys()
                if field not in config or field not in upstream or config[field] != upstream[field]
            )
            if changed:
                mismatches.append((name, changed))
    print(f"{len(local)} GPT-2 configurations checked, {len(mismatches)} mismatches")
    for name, fields in mismatches:
        print(f"{name}: {', '.join(fields)}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
