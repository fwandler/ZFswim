# *ZFswim*
Rate model simulations of larval zebrafish locomotor circuits developed for \[1\]. Three models are available with different numbers of neural populations ("1pop", "2pop", and "8pop"). See \[1\] for the details of these models.

### Requirements
ZFswim requires `numpy` and `scipy`.

### Running a simulation
An example script to run the simulation is included here as `example_run.py`. Simulations use the `snet.py`, `connectome.py`, and `parameters.py` scripts. We recommend not changing these files directly, instead parameters can be changed within the `example_run.py` script.

### Analysis
The file `analysis_functions.py` includes our methods for calculating frequency, amplitude and phase from a completed simulation. This can be imported for your own analysis, or it can be called using `python3 analysis/analysis.py -f {filename}` to calculate and print out the average frequency, amplitude, left-right phase differences and intersegmental phase differences for the simulation output stored at `filename`.

### References
\[1\] Wandler F David, Lemberger Benjamin K, McLean David L, Murray James M (2025) **Coordinated spinal locomotor network dynamics emerge from cell-type-specific connectivity patterns.** *eLife* 14:RP106658 [https://elifesciences.org/articles/106658]
