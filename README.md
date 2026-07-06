# ModalSolver — Mode Basis Neural Network (MBNN)

A physics-aided data-driven **normal-mode ocean acoustic propagation model**.
The MBNN encodes normal-mode theory (WKB approximation) in the structure of a
trainable model, so it can estimate low-frequency acoustic fields from a small
number of measurements, extrapolate beyond the measurement region, and invert
for environmental parameters such as the sound speed profile (SSP).

Based on the paper: K. Li and M. Chitre, *"Physics-aided Data-driven Modal
Ocean Acoustic Propagation Modeling"*, International Congress of Acoustics, 2022.

## Installation

```julia
julia> ]
pkg> add CSV DataFrames Plots ForwardDiff
```

Then place `ModalSolver.jl` in your working folder (proper package
installation coming once this is registered).

## Usage

### 1. Prepare your data

You need acoustic field measurements — ranges (m), depths (m, positive down)
and pressure amplitudes (linear units, e.g. mVpp). Load them however you like
and pass the columns in as vectors. For example, from a CSV:

```julia
using CSV, DataFrames
df = CSV.read("my_measurements.csv", DataFrame)   # columns: range_m, depth_m, amp
```

Alternatively, generate synthetic measurements with a conventional model such
as Kraken (via [AcousticsToolbox.jl](https://github.com/org-arl/AcousticsToolbox.jl))
and use those as training data.

### 2. Train — with or without a known SSP

```julia
include("ModalSolver.jl")
using .ModalSolver
```

**Case A — you know the sound speed profile** (e.g. from a CTD cast). Pass it
in and it is used directly; the model learns only the mode parameters
{A, B, kr}:

```julia
ssp = CSV.read("my_ssp.csv", DataFrame)           # columns: depth_m, c_ms

res = train_mbnn(df.range_m, df.depth_m, df.amp;
                 ssp = (ssp.depth_m, ssp.c_ms),   # or any function z -> c
                 D = 25.0,                        # water depth (m)
                 f = 500.0)                       # source frequency (Hz)
```

**Case B — you don't know the SSP.** Simply omit `ssp`. A small sound speed
neural network (SSNN) is embedded in the model and the SSP is learnt jointly
with the mode parameters, {A, B, kr, SSNN}. You only need a rough prior range
for the sound speed:

```julia
res = train_mbnn(df.range_m, df.depth_m, df.amp;
                 D = 25.0, f = 500.0,
                 cmin = 1445.0, cmax = 1462.0)    # prior SSP bounds (m/s)

c = learned_ssp(res)     # the SSP inverted from the acoustic data
c(12.5)                  # e.g. sound speed at 12.5 m depth
```

Each case has sensible defaults (mode count, learning rate, regularization,
auto-balanced loss weights for Case B), and everything is customizable via
keyword arguments — see `?train_mbnn` in the REPL for the full table.

### 3. Predict anywhere

```julia
amps  = predict_amp(res, [700.0, 710.0], [10.0, 12.0])    # arbitrary points
field = predict_grid(res, 550.0:0.5:800.0, 1.0:0.25:24.0) # full field, incl.
                                                            # extrapolation
active_modes(res)    # learnt modes ranked by |A|+|B|, with wavenumbers kr
```

Complete worked examples reproducing the simulation studies from the paper are
in the [`examples/`](examples/) folder.

## Notes

- Measurements should all come from a single source at a fixed frequency; the
  trained model is valid only at that frequency.
- Depths are positive-down. Ranges are distances from the source.
- Training uses random restarts; results vary slightly between runs. Pass
  `seed_base` for reproducibility, or `seed_base = nothing` for fresh
  randomness each run.
- For SSP inversion with sparse shallow measurements (paper §III-B), train
  Case B and constrain with your known shallow sound speeds — support for a
  `ssp_measurements` keyword is planned.

## References

1. K. Li and M. Chitre, "Physics-aided data-driven modal ocean acoustic
   propagation modeling," International Congress of Acoustics, 2022.
2. K. Li and M. Chitre, "Data-aided underwater acoustic ray propagation
   modeling," IEEE Journal of Oceanic Engineering, 2023.
