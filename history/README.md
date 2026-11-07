# Original development history

`development.bundle` preserves the original 11 local milestone commits, their
original IDs, timestamps and trees, ending at `5b96b10`.

GitHub publication replays those snapshots through the connected API. The API
assigns new commit IDs and publication timestamps. No development dates were
backdated or altered. Recover the original objects used by exported evidence:

```bash
git bundle verify history/development.bundle
git fetch history/development.bundle main:development-history
```

Then the Git references in `evidence/*.json` are available to DVC provenance tools.
New training runs record the current published Git commit normally.
