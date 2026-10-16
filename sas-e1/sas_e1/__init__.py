"""SAS-E1: representation complexity study for steering operators."""
import os

# hf-xet background download threads can race interpreter shutdown ("Bad file
# descriptor" / "PyGILState_Release" fatal error after all work completes).
# Disable Xet transfer unless the user explicitly overrides this.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

__version__ = "0.1.0"
