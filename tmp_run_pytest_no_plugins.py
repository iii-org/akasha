import os
import sys

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
