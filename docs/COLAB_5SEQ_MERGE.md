# Colab run: five missing sequences + merge

Run the setup, helper, and runner cells from `legacy/notebooks/AC_MOT_v10.ipynb` in a T4 Colab session. Then paste/run [`notebooks/AC_MOT_v10_5seq_merge_colab.py`](../notebooks/AC_MOT_v10_5seq_merge_colab.py) cell by cell.

Before Cell 2, set `DRIVE_RESULTS` to the folder containing the preserved 12-sequence per-sequence CSVs, or edit `OLD_12_PER_SEQUENCE_CSVS` in Cell 1 to their exact Drive paths.

The code runs only these five sequences:

`uav0000073_04464_v`, `uav0000120_04775_v`, `uav0000161_00000_v`, `uav0000297_02761_v`, `uav0000370_00001_v`.

It writes new five-sequence output, a merged 17-sequence per-sequence CSV, a 17-sequence summary, and a manifest. It refuses missing inputs, overlapping sequences, duplicate system/sequence rows, or an incomplete 12-sequence input. Existing CSVs are never overwritten.
