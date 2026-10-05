"""Deterministically write TeX/TikZ vector figure fragments from frozen values.

The fragments intentionally contain no experimental computation. Compile main.tex
after this script to render the figures as vector PDF in the manuscript.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

FIGURES = {
"fig_concept.tex": r'''\begin{tikzpicture}[font=\small, node distance=9mm and 9mm, box/.style={draw, rounded corners, align=center, minimum height=10mm, inner sep=3pt}, arr/.style={->,thick}]
\node[box] (scene) {Changing UAV scene\\density, scale, light, blur};
\node[box, right=of scene] (fixed) {Fixed detector\\operating point};
\node[box, below=of fixed] (loss) {Quality/cost mismatch};
\node[box, below=of scene] (sci) {AC--MOT\\scene analysis};
\node[box, right=of sci] (action) {Confidence, NMS,\\resolution action};
\node[box, right=of action] (mot) {Detector $\rightarrow$ tracker\\tracks};
\draw[arr] (scene)--(fixed); \draw[arr] (fixed)--(loss);
\draw[arr] (scene)--(sci); \draw[arr] (sci)--(action); \draw[arr] (action)--(mot);
\end{tikzpicture}''',
"fig_architecture.tex": r'''\begin{tikzpicture}[font=\scriptsize,node distance=5mm and 6mm,box/.style={draw,rounded corners,align=center,minimum height=8mm,inner sep=2.5pt},arr/.style={->,thick}]
\node[box] (frame) {Frame $I_t$}; \node[box,right=of frame] (cue) {Five scene cues};
\node[box,right=of cue] (sci) {SCI + temporal\\smoothing}; \node[box,right=of sci] (cal) {Smart\\Calibrator};
\node[box,right=of cal] (det) {Detector}; \node[box,right=of det] (trk) {Tracker};
\node[box,below=9mm of cal] (ctrl) {confidence / NMS / resolution};
\node[box,below=9mm of trk] (past) {Past tracks only};
\draw[arr](frame)--(cue);\draw[arr](cue)--(sci);\draw[arr](sci)--(cal);\draw[arr](cal)--(ctrl);\draw[arr](ctrl)--(det);\draw[arr](det)--(trk);\draw[arr](trk)--(past);\draw[arr](past.west)-|(cue.south);
\end{tikzpicture}''',
"fig_timeline.tex": r'''\begin{tikzpicture}[font=\scriptsize, node distance=4mm, every node/.style={draw,rounded corners,align=center,inner sep=3pt}, arr/.style={->,thick}]
\node (a) {Original\\five-cue AC--MOT}; \node[right=of a] (b) {Component\\ablations};\node[right=of b] (c) {V1 quality\\calibration};\node[right=of c] (d) {V2 identity/\\speed trade-off};\node[right=of d] (e) {Modern audit\\and v3 freeze};\node[below=of e] (f) {Two hosts, T4\\and UAVDT};
\draw[arr](a)--(b);\draw[arr](b)--(c);\draw[arr](c)--(d);\draw[arr](d)--(e);\draw[arr](e)--(f);
\end{tikzpicture}''',
"fig_runtime.tex": r'''\begin{tikzpicture}[x=0.055cm,y=0.45cm,font=\scriptsize]
\draw[->] (0,0)--(160,0) node[right]{mean latency (ms)};
\foreach \x in {0,40,80,120,160} \draw (\x,0) -- (\x,-.12) node[below]{\x};
\node[left] at (0,1) {ByteTrack}; \fill[blue!60] (0,.65) rectangle (74.3,1.35); \node[right] at (74.3,1) {74.3};
\node[left] at (0,2) {OATrack}; \fill[blue!60] (0,1.65) rectangle (65.5,2.35); \node[right] at (65.5,2) {65.5};
\node[left] at (0,3) {Baseline ByteTrack}; \fill[gray!60] (0,2.65) rectangle (139.9,3.35); \node[right] at (139.9,3) {139.9};
\end{tikzpicture}'''
}

for name, content in FIGURES.items():
    (OUT / name).write_text(content + "\n", encoding="utf-8")
print(f"wrote {len(FIGURES)} TikZ fragments to {OUT}")
