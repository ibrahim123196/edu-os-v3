"""Settings for the whole test run.

The tests call the API without logging in -- most were written before
accounts -- so login is switched off for them, and ACTIVE_STUDENT decides
who they are, as it did before. Set here, before anything reads the
settings. The tests of logging in switch it back on for themselves
(tests/integration/test_accounts_sql.py).
"""

import os

os.environ["REQUIRE_LOGIN"] = "false"
# Likewise the limits on AI use: the tests ask far more than a student may.
# tests/integration/test_ai_limits_sql.py switches them on for itself.
for name in ("AI_REQUESTS_PER_HOUR", "AI_REQUESTS_PER_DAY", "AI_DAILY_BUDGET_USD"):
    os.environ[name] = "0"
