"""Download the revision-pinned FlexiBERT and HAT evaluation inputs."""

from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import re


DATA = Path(__file__).resolve().parent / "data"
FLEXIBERT = (
    "https://raw.githubusercontent.com/aaronserianni/training-free-nas/"
    "2d76e01b9586cad7340e8268dadba3056efd070b/data/BERT_benchmark.json"
)
FLEXIBERT_SHA256 = "fa2ff7198dcd71e0b8021ed2ace1c221672c33cc56931b6dff31fda3d044784f"
HAT = (
    "https://codeload.github.com/mit-han-lab/hardware-aware-transformers/"
    "zip/70e5a279d080670208249fdd98ed731fa9bcc466"
)


def download(url):
    """Read an upstream artifact over HTTPS with a bounded connection timeout."""
    with urlopen(url, timeout=60) as response:
        return response.read()


def main():
    """Save FlexiBERT and only the BLEU-labeled HAT configurations."""
    bert = download(FLEXIBERT)
    if hashlib.sha256(bert).hexdigest() != FLEXIBERT_SHA256:
        raise ValueError("FlexiBERT benchmark SHA-256 mismatch")

    # Write only known paths beneath data/hat, never extract the whole archive.
    with ZipFile(BytesIO(download(HAT))) as archive:
        configs = []
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) != 5 or parts[1] != "configs" or parts[3] != "subtransformer":
                continue
            if not re.fullmatch(r"[a-z0-9.-]+", parts[2]):
                continue
            if not re.fullmatch(r"HAT_[^/]+_bleu@[0-9.]+\.yml", parts[4]):
                continue
            configs.append((parts[2], parts[4], archive.read(name)))
    if not configs:
        raise ValueError("No BLEU-labeled HAT configurations found")

    (DATA / "BERT_benchmark.json").write_bytes(bert)
    for task, filename, payload in configs:
        directory = DATA / "hat" / task / "subtransformer"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(payload)
    print(f"Prepared FlexiBERT data and {len(configs)} BLEU-labeled HAT files in {DATA}")


if __name__ == "__main__":
    main()
