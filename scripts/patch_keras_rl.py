"""Patch keras-rl2 so it imports from tf_keras.

keras-rl2 (last released 2020) imports Keras from internal TensorFlow paths
that newer TensorFlow versions removed. Run this once after `pip install`:

    python scripts/patch_keras_rl.py
"""
import importlib.util
import pathlib

REPLACEMENTS = [
    ("from tensorflow.keras import __version__ as KERAS_VERSION",
     "from tf_keras import __version__ as KERAS_VERSION"),
    ("from tensorflow.python.keras.utils.generic_utils import Progbar",
     "from tf_keras.utils import Progbar"),
    ("tensorflow.python.keras", "tf_keras"),
]


def main():
    spec = importlib.util.find_spec("rl")
    if spec is None:
        raise SystemExit("keras-rl2 is not installed. Run: pip install -r requirements.txt")
    rl_dir = pathlib.Path(spec.submodule_search_locations[0])

    patched = 0
    for f in rl_dir.rglob("*.py"):
        text = f.read_text()
        new = text
        for old, rep in REPLACEMENTS:
            new = new.replace(old, rep)
        if new != text:
            f.write_text(new)
            patched += 1
            print("patched:", f.relative_to(rl_dir.parent))
    print(f"done ({patched} file(s) changed)")


if __name__ == "__main__":
    main()
