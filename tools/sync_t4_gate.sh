#!/bin/zsh
# Sync T4 gate caches + benchmark from Drive (no evaluation of transfer sets).
cd /Users/ahmedgouda/Desktop/Universal-ACMOT
SRC="/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@gmail.com/My Drive/Universal-ACMOT-Results/t4_gate"
until [ -f "$SRC/T4_GATE_DONE.json" ]; do sleep 60; done; sleep 60
mkdir -p outputs/det_cache_train_t4/yolov8 outputs/det_cache_train_t4/rtdetr outputs/t4_gate
cp "$SRC/train/yolov8/"*.npz outputs/det_cache_train_t4/yolov8/
cp "$SRC/train/rtdetr/"*.npz outputs/det_cache_train_t4/rtdetr/
cp -R outputs/det_cache_train/visual_cues outputs/det_cache_train_t4/ 2>/dev/null
cp "$SRC"/*.json "$SRC"/packages.txt outputs/t4_gate/ 2>/dev/null
mkdir -p outputs/t4_gate/val_fasterrcnn && cp "$SRC/val/fasterrcnn/"*.npz outputs/t4_gate/val_fasterrcnn/
echo "T4 GATE SYNCED $(date +%T)"
