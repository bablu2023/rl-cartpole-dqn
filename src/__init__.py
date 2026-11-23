"""CartPole DQN project package.

keras-rl2 only works with Keras 2, so we tell TensorFlow to use the
tf-keras package (Keras 2) instead of Keras 3. This must happen before
TensorFlow is imported, which is why it lives in the package __init__:
any `import src...` runs it first.
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
