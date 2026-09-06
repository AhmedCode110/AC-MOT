# AC-MOT Versioning Rule

This rule is mandatory for future research changes.

## Immutable-version policy

1. Never overwrite the archived implementation of an existing numbered version.
2. Any requested code/research/protocol change after a numbered version creates the next integer version: v14 -> v15 -> v16 -> ...
3. The previous version must remain reproducible through its archived files and/or immutable Git blob/commit references.
4. The root/current runner may point to the latest version, but archived version folders are historical records and must not be repurposed.
5. Every version must have `versions/vNN/README.md`.

## Required README sections

Each version README must state:

- starting version and previous verified output
- exact problem discovered in the previous version
- exact files/code/protocol changed
- why each change was made
- what was intentionally kept unchanged
- timing/FPS definition if performance is involved
- expected output before verification
- actual verified output after the run is available
- remaining problem, if any
- why the next version is needed, if another version is created

## Realtime result rule

Never label a version realtime based only on an intended optimization. `PASS_REALTIME` requires a saved measured result at or above the configured FPS target under that version's documented timing protocol.

## Research-integrity rule

Do not silently convert a fixed baseline into an adaptive one, do not silently change detector/tracker thresholds, precision, frame source, inference count, or timing inclusion/exclusion. Any such change must be versioned and documented.
