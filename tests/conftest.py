import os
import sys

# Settings reads these at import time, so give them harmless test values.
for name in [
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "JIGSAWSTACK_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
]:
    os.environ.setdefault(name, "test-value")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
