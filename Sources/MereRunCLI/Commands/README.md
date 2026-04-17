# mere.run CLI Commands

This directory owns the public CLI surface.

- one command file per modality or command cluster
- parser defaults and public flags are covered in `Tests/MereRunCLITests/`
- stdout should remain machine-readable where possible; diagnostics belong on stderr

If you change a flag, subcommand name, or help contract, update the nearest parsing tests and any affected user-facing docs in the same change.
