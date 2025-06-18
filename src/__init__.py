import os
import sys
import importlib.machinery
from types import ModuleType
from unittest.mock import MagicMock

if os.getenv("PDOC_MOCK_TF") == "1":
    tf_mock = ModuleType("tensorflow")
    tf_mock.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
    tf_mock.math = MagicMock()
    tf_mock.math.log = lambda x: x  # Return input for docgen safety

    sys.modules["tensorflow"] = tf_mock
    sys.modules["tensorflow.math"] = tf_mock.math
