import json

import numpy as np
from PIL import Image
import pytest

from scripts.extract_omega_cache import load_manifest


def test_load_manifest_resolves_paths_relative_to_manifest(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for camera in ("agentview_left", "agentview_right", "eye_in_hand"):
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_dir / f"{camera}.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "episode": 0,
                "step": 0,
                "images": {
                    camera: f"images/{camera}.png" for camera in ("agentview_left", "agentview_right", "eye_in_hand")
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_manifest(manifest, 1, ("agentview_left", "agentview_right", "eye_in_hand"))
    assert all(path.is_absolute() for path in records[0]["paths"])


def test_load_manifest_rejects_missing_view(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"episode": 0, "step": 0, "images": {"agentview_left": "left.png"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid manifest"):
        load_manifest(manifest, 1, ("agentview_left", "agentview_right", "eye_in_hand"))
