# Public copies of the re-scoring outputs

`rescore/` holds the outputs exactly as the tool wrote them (commit 97853c6) and stays unchanged; it is the provenance copy.
This folder is a copy for public release with one substitution:

- every occurrence of the local Google Drive mount prefix `/Users/<user>/Library/CloudStorage/GoogleDrive-<account>/My Drive/` is replaced by `<DRIVE>/`.

No other byte differs; numbers, hashes of the scored input files and reproduction checks are unchanged. Files with 0 substitutions are byte-identical to the originals.

Git history of this repository still contains the original paths (commit 97853c6 and later). A public release must therefore be a fresh export (new repository or archive built from these public copies), not a push of this history.

| file | sha256 original (`rescore/`) | sha256 public (`rescore_public/`) | substitutions |
|---|---|---|---|
| `sparsetrack/sparsetrack_adaptive_per_sequence.csv` | `3e47bfc83b04f38ce9607535703bdc12ba43f82f52ff0efe3994aadbea4e77a7` | `3e47bfc83b04f38ce9607535703bdc12ba43f82f52ff0efe3994aadbea4e77a7` | 0 |
| `sparsetrack/sparsetrack_static070_per_sequence.csv` | `34cfaee06d4c9b4985b528cf9af3fc8be5b96913cdf780f871601f8fbdddbf28` | `34cfaee06d4c9b4985b528cf9af3fc8be5b96913cdf780f871601f8fbdddbf28` | 0 |
| `sparsetrack/sparsetrack_static075_per_sequence.csv` | `278899c57f107107bab1f7c2e1c178ee43d46c1c44cd125bc6cbf874ba289cb0` | `278899c57f107107bab1f7c2e1c178ee43d46c1c44cd125bc6cbf874ba289cb0` | 0 |
| `sparsetrack/sparsetrack_static080_per_sequence.csv` | `c7bdcffb7eabcabedf355c3121afe13436db716994c32bcd555a3098d830ab60` | `c7bdcffb7eabcabedf355c3121afe13436db716994c32bcd555a3098d830ab60` | 0 |
| `sparsetrack/sparsetrack_val_half_rescore.json` | `09cfc55cf5e8a797b58e649b8527d52b1b229cfab303333aeb1adc55114b8059` | `02c6c4f1b2c31c0f8382fbb7d38498f04d6ed1dd28a4042cc10a53d1a7390128` | 8 |
| `sparsetrack_console.txt` | `7baf03b7ee858357b0ed4bc735040e3b329876fd66ac6b6cfe7ea8f791c8efa2` | `7baf03b7ee858357b0ed4bc735040e3b329876fd66ac6b6cfe7ea8f791c8efa2` | 0 |
| `u2mot/u2mot_baseline_per_sequence.csv` | `4ba2dacf1ad28c1f9d3f8f03770ec5049352d0196aa6fe66b3f27c857402f4ed` | `4ba2dacf1ad28c1f9d3f8f03770ec5049352d0196aa6fe66b3f27c857402f4ed` | 0 |
| `u2mot/u2mot_controller_per_sequence.csv` | `9ccd6587aebb834c63499691704775eb78a0e4583e336a6a2b635c5efd7ded4f` | `9ccd6587aebb834c63499691704775eb78a0e4583e336a6a2b635c5efd7ded4f` | 0 |
| `u2mot/u2mot_testdev_rescore.json` | `b654daa012178b6e99740809148157180829b0563647054d720705a83c9ba828` | `b654daa012178b6e99740809148157180829b0563647054d720705a83c9ba828` | 0 |
| `u2mot_console.txt` | `f21a985b50592fdf5f476584f624689e213a2e369bdbff0fe68c52eab564bdb5` | `f21a985b50592fdf5f476584f624689e213a2e369bdbff0fe68c52eab564bdb5` | 0 |
