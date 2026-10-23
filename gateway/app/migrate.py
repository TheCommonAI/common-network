"""Nothing to migrate: the retired gateway has no database.

The Railway service still runs `python -m app.migrate` before each deploy (a pre-deploy command in the
service settings). Without this module that command fails and every deploy is rolled back. It can be
deleted once the pre-deploy command is cleared in the Railway settings.
"""
if __name__ == "__main__":
    print("retired gateway: no database, nothing to migrate")
