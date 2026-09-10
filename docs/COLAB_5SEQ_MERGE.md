# Colab run: five missing sequences + merge

Run the setup, helper, and runner cells from `legacy/notebooks/AC_MOT_v10.ipynb` in a T4 Colab session. Then paste/run [`notebooks/AC_MOT_v10_5seq_merge_colab.py`](../notebooks/AC_MOT_v10_5seq_merge_colab.py) cell by cell.

All generated artifacts are saved under `/content/drive/MyDrive/concept experiment base 17 sequence`. The preserved 12-sequence CSVs may remain anywhere under MyDrive; the notebook discovers them by their exact filenames.

The code runs only these five sequences:

`uav0000073_04464_v`, `uav0000120_04775_v`, `uav0000161_00000_v`, `uav0000297_02761_v`, `uav0000370_00001_v`.

It writes the new five-sequence output, merged 17-sequence CSVs, manifest, Excel workbook, and comparison chart into that folder. It refuses missing inputs, overlapping sequences, duplicate system/sequence rows, or an incomplete 12-sequence input. Existing CSVs are never overwritten.
