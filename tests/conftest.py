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

# supabase-py validates the key format, so use a JWT shaped placeholder.
os.environ["SUPABASE_ANON_KEY"] = os.environ["SUPABASE_SERVICE_KEY"] = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.c2lnbmF0dXJl"
)

# The Supabase client checks that the url looks like a url.
os.environ["SUPABASE_URL"] = "https://example.supabase.co"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
