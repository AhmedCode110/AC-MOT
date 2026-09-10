# AC-MOT legacy v10 — run the five missing sequences, then merge with the old 12.
#
# Use this after running the setup/helper/runner cells from:
# legacy/notebooks/AC_MOT_v10.ipynb
# It intentionally reuses that notebook's run_system() implementation so the
# five new sequences use the same detector, tracker, SCI, metric and timing
# protocol as the preserved 12-sequence results.

# %% Cell 1 — select only the five missing sequences
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

MISSING_SEQS = [
    'uav0000073_04464_v',
    'uav0000120_04775_v',
    'uav0000161_00000_v',
    'uav0000297_02761_v',
    'uav0000370_00001_v',
]

by_name = {s.name: s for s in all_sequences}
missing_on_disk = [n for n in MISSING_SEQS if n not in by_name]
assert not missing_on_disk, f'Missing dataset folders: {missing_on_disk}'
VAL_SEQS = [by_name[n] for n in MISSING_SEQS]
assert len(VAL_SEQS) == 5 and len({s.name for s in VAL_SEQS}) == 5

# The old 12-sequence per-sequence outputs are required for a valid merge.
# Change these two paths to the exact files in Drive.
OLD_12_PER_SEQUENCE_CSVS = [
    DRIVE_RESULTS / 'FINAL_RUN1_A0A1_20260603_081343_per_seq.csv',
    DRIVE_RESULTS / 'FINAL_RUN2_A2A3_20260603_080537_per_seq.csv',
]

print('Running exactly these five sequences:')
for s in VAL_SEQS:
    print(' -', s.name)

# %% Cell 2 — same four-way ablation, no ReID
ABLATION_SYSTEMS = [
    dict(name='A0_Baseline_Default', model=MODEL_NAME, tracker='baseline',
         adaptive_threshold=False, adaptive_resolution=False,
         scene_analysis=False, reid=False),
    dict(name='A1_TunedTracker', model=MODEL_NAME, tracker='acmot',
         adaptive_threshold=False, adaptive_resolution=False,
         scene_analysis=False, reid=False),
    dict(name='A2_AdaptThreshold', model=MODEL_NAME, tracker='acmot',
         adaptive_threshold=True, adaptive_resolution=False,
         scene_analysis=True, reid=False),
    dict(name='A3_AdaptResolution', model=MODEL_NAME, tracker='acmot',
         adaptive_threshold=True, adaptive_resolution=True,
         scene_analysis=True, reid=False),
]

run_tag = f'acmot_v10_ablation_missing5_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
new_rows = []
for system in ABLATION_SYSTEMS:
    # run_system, TRACKERS, ANNOT_DIR, MODEL_NAME and DRIVE_RESULTS come from
    # the original ready notebook. No old result is overwritten.
    new_rows.append(run_system(system, VAL_SEQS, run_tag))

new_per_sequence = pd.concat(new_rows, ignore_index=True)
assert set(new_per_sequence['sequence']) == set(MISSING_SEQS)
assert set(new_per_sequence['system']) == {
    'A0_Baseline_Default', 'A1_TunedTracker',
    'A2_AdaptThreshold', 'A3_AdaptResolution',
}
new_path = DRIVE_RESULTS / f'{run_tag}_per_sequence.csv'
new_per_sequence.to_csv(new_path, index=False)
print('Saved new five-sequence rows ->', new_path)

# %% Cell 3 — validate, merge and summarize all 17 sequences
old_frames = []
for path in OLD_12_PER_SEQUENCE_CSVS:
    path = Path(path)
    assert path.exists(), f'Old per-sequence CSV not found: {path}'
    old_frames.append(pd.read_csv(path))

old = pd.concat(old_frames, ignore_index=True)
name_map = {
    'A1_TunedTracker_Only': 'A1_TunedTracker',
    'A2_TunedTracker_AdaptThresh': 'A2_AdaptThreshold',
    'A3_TunedTracker_AdaptThresh_Res': 'A3_AdaptResolution',
    'Baseline_Default': 'A0_Baseline_Default',
    'Baseline_TunedTracker': 'A1_TunedTracker',
    'AC-MOT_v10': 'A3_AdaptResolution',
}
old['system'] = old['system'].replace(name_map)
required_systems = {s['name'] for s in ABLATION_SYSTEMS}

old12 = old[old['system'].isin(required_systems)].copy()
assert old12['sequence'].nunique() == 12, sorted(old12['sequence'].unique())
assert not set(old12['sequence']).intersection(MISSING_SEQS)
assert set(old12['system']) == required_systems

new = new_per_sequence.copy()
assert not set(new['sequence']).intersection(set(old12['sequence']))
merged = pd.concat([old12, new], ignore_index=True)
assert merged['sequence'].nunique() == 17
assert set(merged['sequence']) == set(old12['sequence']).union(MISSING_SEQS)
assert not merged.duplicated(['system', 'sequence']).any()
assert set(merged['system']) == required_systems

# Preserve columns shared by both old and new reports, then aggregate exactly
# like the original ablation table: macro means for rates, sums for counts.
summary = []
for system, g in merged.groupby('system', sort=False):
    summary.append(dict(
        system=system,
        sequences=int(g['sequence'].nunique()),
        mota=float(g['mota'].mean()),
        idf1=float(g['idf1'].mean()),
        hota=float(g['hota'].mean()),
        recall=float(g['recall'].mean()),
        precision=float(g['precision'].mean()),
        ids=int(g['ids'].sum()),
        fn=int(g['fn'].sum()),
        fp=int(g['fp'].sum()),
        fps=float(g['fps'].mean()),
    ))
summary = pd.DataFrame(summary)
base = summary.iloc[0]
summary['mota_delta'] = summary['mota'] - float(base['mota'])
summary['idf1_delta'] = summary['idf1'] - float(base['idf1'])
summary['hota_delta'] = summary['hota'] - float(base['hota'])
summary['recall_delta'] = summary['recall'] - float(base['recall'])
summary['precision_delta'] = summary['precision'] - float(base['precision'])
summary['fps_delta'] = summary['fps'] - float(base['fps'])
summary['ids_delta'] = summary['ids'] - int(base['ids'])
summary['ids_reduction'] = int(base['ids']) - summary['ids']
summary['ids_reduction_pct'] = (int(base['ids']) - summary['ids']) / max(1, int(base['ids'])) * 100.0
summary['realtime_20fps'] = summary['fps'] >= 20.0
summary['strict_realtime_25fps'] = summary['fps'] >= 25.0

# Report every objective separately. A single winner is only declared when a
# method is best on the requested metric; this avoids hiding the IDS trade-off.
realtime = summary[summary['realtime_20fps']].copy()
best_realtime = (realtime.sort_values(['mota', 'hota', 'ids', 'fps'],
                                      ascending=[False, False, True, False])
                 .iloc[0]['system'] if len(realtime) else None)
best_mota = summary.loc[summary['mota'].idxmax(), 'system']
best_hota = summary.loc[summary['hota'].idxmax(), 'system']
best_ids = summary.loc[summary['ids'].idxmin(), 'system']
best_fps = summary.loc[summary['fps'].idxmax(), 'system']

def dominates(a, b):
    return (a['mota'] >= b['mota'] and a['hota'] >= b['hota'] and
            a['ids'] <= b['ids'] and a['fps'] >= b['fps'] and
            (a['mota'] > b['mota'] or a['hota'] > b['hota'] or
             a['ids'] < b['ids'] or a['fps'] > b['fps']))

pareto = [a['system'] for _, a in summary.iterrows()
          if not any(dominates(b, a) for _, b in summary.iterrows() if b['system'] != a['system'])]
acmot = summary[summary['system'] == 'A3_AdaptResolution'].iloc[0]
acmot_dominates_all = all(dominates(acmot, b) for _, b in summary.iterrows()
                          if b['system'] != 'A3_AdaptResolution')

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
merged_path = DRIVE_RESULTS / f'acmot_v10_ablation_merged17_{stamp}_per_sequence.csv'
summary_path = DRIVE_RESULTS / f'acmot_v10_ablation_merged17_{stamp}_summary.csv'
manifest_path = DRIVE_RESULTS / f'acmot_v10_ablation_merged17_{stamp}_manifest.json'
excel_path = DRIVE_RESULTS / f'acmot_v10_ablation_merged17_{stamp}_comparison.xlsx'
chart_path = DRIVE_RESULTS / f'acmot_v10_ablation_merged17_{stamp}_comparison.png'
merged.to_csv(merged_path, index=False)
summary.to_csv(summary_path, index=False)
Path(manifest_path).write_text(json.dumps({
    'protocol': 'AC-MOT legacy v10 four-way ablation',
    'old_sequence_count': 12,
    'new_sequence_count': 5,
    'merged_sequence_count': 17,
    'new_sequences': MISSING_SEQS,
    'old_input_csvs': [str(p) for p in OLD_12_PER_SEQUENCE_CSVS],
    'new_input_csv': str(new_path),
    'duplicate_check': 'passed',
    'metric_note': 'macro means for rates; sums for IDS/FN/FP; no old input overwritten',
    'realtime_policy': 'acceptable realtime is >=20 FPS; strict realtime is >=25 FPS',
    'best_realtime_20fps': best_realtime,
    'best_mota': best_mota,
    'best_hota': best_hota,
    'lowest_ids': best_ids,
    'best_fps': best_fps,
    'pareto_frontier': pareto,
    'acmot_dominates_all_objectives': bool(acmot_dominates_all),
}, indent=2) + '\n')

# Create a reproducible Excel handoff from the real 17-sequence outputs.
# This does not invent or overwrite metrics; it only formats `merged`/`summary`.
import matplotlib.pyplot as plt

parameter_rows = [
    dict(system='A0_Baseline_Default', tracker='baseline', adaptive_threshold=False,
         adaptive_resolution=False, scene_analysis=False, reid=False,
         protocol_note='Default baseline; fixed detector/tracker settings'),
    dict(system='A1_TunedTracker', tracker='acmot', adaptive_threshold=False,
         adaptive_resolution=False, scene_analysis=False, reid=False,
         protocol_note='Tuned tracker only'),
    dict(system='A2_AdaptThreshold', tracker='acmot', adaptive_threshold=True,
         adaptive_resolution=False, scene_analysis=True, reid=False,
         protocol_note='Tuned tracker + adaptive threshold/scene analysis'),
    dict(system='A3_AdaptResolution', tracker='acmot', adaptive_threshold=True,
         adaptive_resolution=True, scene_analysis=True, reid=False,
         protocol_note='Full AC-MOT: adaptive threshold + adaptive resolution'),
]
parameters = pd.DataFrame(parameter_rows)
metric_cols = ['mota', 'idf1', 'hota', 'recall', 'precision', 'fps', 'ids']
step_effect = summary[['system'] + metric_cols].copy()
step_effect.insert(1, 'previous_system', ['—'] + summary['system'].iloc[:-1].tolist())
for metric in metric_cols:
    step_effect[f'{metric}_change_vs_previous'] = summary[metric].diff()
winners = pd.DataFrame([
    {'objective': 'Best acceptable realtime (>=20 FPS)', 'winner': best_realtime or 'NONE'},
    {'objective': 'Best strict realtime (>=25 FPS)', 'winner': (summary[summary['strict_realtime_25fps']]
        .sort_values(['mota', 'hota', 'ids', 'fps'], ascending=[False, False, True, False])
        .iloc[0]['system'] if summary['strict_realtime_25fps'].any() else 'NONE')},
    {'objective': 'Best MOTA', 'winner': best_mota},
    {'objective': 'Best HOTA', 'winner': best_hota},
    {'objective': 'Lowest IDS', 'winner': best_ids},
    {'objective': 'Best FPS', 'winner': best_fps},
    {'objective': 'Pareto frontier', 'winner': ', '.join(pareto)},
    {'objective': 'AC-MOT dominates all objectives', 'winner': bool(acmot_dominates_all)},
])

with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    summary.to_excel(writer, sheet_name='Summary', index=False)
    merged.to_excel(writer, sheet_name='Per_Sequence', index=False)
    parameters.to_excel(writer, sheet_name='Parameters', index=False)
    step_effect.to_excel(writer, sheet_name='Step_Effects', index=False)
    winners.to_excel(writer, sheet_name='Objective_Winners', index=False)
    manifest = pd.DataFrame([{
        'merged_csv': str(merged_path), 'summary_csv': str(summary_path),
        'manifest_json': str(manifest_path), 'sequence_count': int(merged['sequence'].nunique()),
        'systems': ', '.join(required_systems), 'metric_protocol': 'macro mean rates; sum IDS/FN/FP',
    }])
    manifest.to_excel(writer, sheet_name='Manifest', index=False)
    workbook = writer.book
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#1F4E78', 'font_color': 'white'})
    for sheet_name, frame in [('Summary', summary), ('Per_Sequence', merged),
                              ('Parameters', parameters), ('Step_Effects', step_effect), ('Objective_Winners', winners),
                              ('Manifest', manifest)]:
        sheet = writer.sheets[sheet_name]
        sheet.freeze_panes(1, 0)
        sheet.autofilter(0, 0, len(frame), max(0, len(frame.columns) - 1))
        sheet.set_column(0, max(0, len(frame.columns) - 1), 18)
        for col_idx, col_name in enumerate(frame.columns):
            sheet.write(0, col_idx, col_name, header_fmt)
    for col in ['mota', 'idf1', 'hota', 'recall', 'precision', 'mota_delta']:
        if col in summary:
            idx = summary.columns.get_loc(col)
            writer.sheets['Summary'].set_column(idx, idx, 14, workbook.add_format({'num_format': '0.0000'}))
    chart_sheet = workbook.add_worksheet('Charts')
    chart_sheet.write('A1', '17-sequence AC-MOT ablation comparison')
    chart_sheet.write('A2', 'Charts use the actual merged summary generated in this run.')
    chart_sheet.insert_chart('A4', {'type': 'column', 'subtype': 'clustered',
        'title': {'name': 'MOTA / HOTA'}, 'y_axis': {'name': 'score'},
        'categories': "='Summary'!$A$2:$A$5",
        'series': [{'name': 'MOTA', 'values': "='Summary'!$C$2:$C$5"},
                   {'name': 'HOTA', 'values': "='Summary'!$E$2:$E$5"}]})
    chart_sheet.insert_chart('J4', {'type': 'column', 'subtype': 'clustered',
        'title': {'name': 'FPS / IDS'}, 'y_axis': {'name': 'value'},
        'categories': "='Summary'!$A$2:$A$5",
        'series': [{'name': 'FPS', 'values': "='Summary'!$K$2:$K$5"},
                   {'name': 'IDS', 'values': "='Summary'!$H$2:$H$5"}]})

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, metric, title in zip(axes, ['mota', 'hota', 'ids', 'fps'], ['MOTA', 'HOTA', 'IDS', 'FPS']):
    values = summary[metric]
    ax.bar(summary['system'], values, color=['#777777', '#4C78A8', '#F58518', '#54A24B'])
    ax.set_title(title); ax.tick_params(axis='x', rotation=35); ax.grid(axis='y', alpha=.25)
fig.tight_layout(); fig.savefig(chart_path, dpi=180, bbox_inches='tight'); plt.close(fig)

print('\nMERGED 17-SEQUENCE ABLATION')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('\nOBJECTIVE WINNERS')
print('Best acceptable realtime (>=20 FPS):', best_realtime or 'NONE')
print('Best MOTA:', best_mota)
print('Best HOTA:', best_hota)
print('Lowest IDS:', best_ids)
print('Best FPS:', best_fps)
print('Pareto frontier:', ', '.join(pareto))
print('AC-MOT dominates all objectives:', acmot_dominates_all)
print('\nSaved:')
print(merged_path)
print(summary_path)
print(manifest_path)
print(excel_path)
print(chart_path)
