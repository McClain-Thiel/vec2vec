"""Project settings.

See https://docs.kedro.org/en/stable/configure/configuration_basics/ for the
full set of options and their defaults.
"""

from pathlib import Path

from dotenv import load_dotenv
from omegaconf.resolvers import oc

# Load local developer credentials before OmegaConf resolves environment values.
# An explicit shell, CI, or job-runner value remains authoritative.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# Keyword arguments passed to the configuration loader.
#
# ``globals.yml`` holds the storage roots every catalog entry is built from, and
# the ``oc.env`` resolver lets ``parameters.yml`` read the OpenRouter key from
# the environment instead of keeping a secret in a file.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "globals": ["globals*", "globals*/**"],
    },
    "custom_resolvers": {
        "oc.env": oc.env,
    },
}
