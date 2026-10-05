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
"fig_runtime.tex": r'''\begin{tikzpicture}[font=\scriptsize, x=0.085cm, y=0.55cm]
\draw[->] (0,0)--(180,0) node[right]{mean latency (ms)};
\foreach \x in {0,40,80,120,160} {\draw (\x,0)--(\x,-.12) node[below]{\x};}
\node[anchor=east] at (-2,1) {baseline B}; \fill[gray!55] (0,.65) rectangle (139.9,1.35); \node[anchor=west] at (142,1) {139.9};
\node[anchor=east] at (-2,2) {v3 B}; \fill[blue!65] (0,1.65) rectangle (74.3,2.35); \node[anchor=west] at (76.5,2) {74.3};
\node[anchor=east] at (-2,3) {baseline O}; \fill[gray!55] (0,2.65) rectangle (128.9,3.35); \node[anchor=west] at (131,3) {128.9};
\node[anchor=east] at (-2,4) {v3 O}; \fill[blue!65] (0,3.65) rectangle (65.5,4.35); \node[anchor=west] at (67.5,4) {65.5};
\node[draw, fill=gray!55, minimum width=8mm, minimum height=3mm] at (20,5.2) {}; \node[anchor=west] at (28,5.2) {declared baseline};
\node[draw, fill=blue!65, minimum width=8mm, minimum height=3mm] at (100,5.2) {}; \node[anchor=west] at (108,5.2) {v3};
\end{tikzpicture}''',
"fig_state_machine.tex": r'''\begin{tikzpicture}[font=\scriptsize,node distance=5mm and 6mm,box/.style={draw,rounded corners,align=center,minimum height=8mm,text width=22mm,inner sep=2pt},wide/.style={draw,rounded corners,align=center,minimum height=8mm,text width=27mm,inner sep=2pt},arr/.style={->,thick},feedback/.style={->,dashed,thick}]
\node[box] (input) {Current image\\prior tracker boxes}; \node[box,right=of input] (cues) {Image cues +\\crowd/tiny};
\node[wide,right=of cues] (sci) {Seven-entry\\crowd SCI history}; \node[wide,right=of sci] (target) {Target state\\HIGH / MEDIUM / LOW};
\node[wide,right=of target] (guards) {Recovery probe,\\hysteresis, 30-frame dwell}; \node[wide,right=of guards] (action) {Frozen action\\resolution / NMS / confidence};
\node[box,right=of action] (track) {Detector +\\tracker};
\draw[arr](input)--(cues); \draw[arr](cues)--(sci); \draw[arr](sci)--(target); \draw[arr](target)--(guards); \draw[arr](guards)--(action); \draw[arr](action)--(track);
\draw[feedback](track.south) |- ++(0,-8mm) -| (input.south);
\node[anchor=north] at ($(track.south)!0.5!(input.south)+(0,-8mm)$) {feedback is causal and delayed};
\end{tikzpicture}''',
"fig_modern_results.tex": r'''\begin{tikzpicture}[font=\scriptsize,x=0.025cm,y=0.10cm]
\draw[->](0,0)--(520,0) node[right]{system}; \draw[->](0,0)--(0,55) node[above]{HOTA};
\foreach \y in {0,10,20,30,40,50} {\draw(-2,\y)--(0,\y); \node[anchor=east] at(-3,\y){\y};}
\node[anchor=north] at(95,-4){ByteTrack}; \node[anchor=north] at(325,-4){OATrack};
\fill[gray!55](55,0) rectangle(105,41.586); \fill[blue!65](115,0) rectangle(165,44.523);
\fill[gray!55](285,0) rectangle(335,46.770); \fill[blue!65](345,0) rectangle(395,48.095);
\node[anchor=south] at(80,41.586){41.59}; \node[anchor=south] at(140,44.523){44.52}; \node[anchor=south] at(310,46.770){46.77}; \node[anchor=south] at(370,48.095){48.10};
\node[draw,fill=gray!55,minimum width=8mm,minimum height=3mm] at(430,42){}; \node[anchor=west] at(438,42){baseline};
\node[draw,fill=blue!65,minimum width=8mm,minimum height=3mm] at(430,36){}; \node[anchor=west] at(438,36){v3};
\end{tikzpicture}''',
"fig_uavdt.tex": r'''\begin{tikzpicture}[font=\scriptsize,x=0.35cm,y=0.65cm]
\draw[->](-4,0)--(30,0) node[right]{MOTA}; \foreach \x in {0,5,10,15,20,25} {\draw(\x,0)--(\x,-.10) node[below]{\x};}
\node[anchor=east] at(-4,1){ByteTrack baseline}; \fill[gray!55](0,.65) rectangle(-1.57,1.35); \node[anchor=west] at(0.3,1){-1.57};
\node[anchor=east] at(-4,2){ByteTrack v3}; \fill[blue!65](0,1.65) rectangle(16.03,2.35); \node[anchor=west] at(17,2){16.03};
\node[anchor=east] at(-4,3){OATrack baseline}; \fill[gray!55](0,2.65) rectangle(19.61,3.35); \node[anchor=west] at(20.5,3){19.61};
\node[anchor=east] at(-4,4){OATrack v3}; \fill[blue!65](0,3.65) rectangle(22.87,4.35); \node[anchor=west] at(23.8,4){22.87};
\end{tikzpicture}'''
}

for name, content in FIGURES.items():
    (OUT / name).write_text(content + "\n", encoding="utf-8")
print(f"wrote {len(FIGURES)} TikZ fragments to {OUT}")
