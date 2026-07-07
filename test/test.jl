include("ModalSolver.jl"); using .ModalSolver
using Statistics, Plots, CSV, DataFrames

ssp = load_ssp("true_ssnn_ssp.csv")
pm  = ModeSolver(D=25.0, f=500.0, ssp=ssp)
fit!(pm, "ssnn_profiles_1224.csv")

# --- field heatmap (shows in the Julia Plots pane, and saves a PNG) ---
ranges = 550.0:0.5:800.0
depths = 1.0:0.25:24.0
field  = predict_grid(pm, ranges, depths)

plt = heatmap(ranges, depths, 20 .* log10.(field); yflip=true,
              xlabel="Range (m)", ylabel="Depth (m)",
              title="ModalSolver known-SSP (reproduction)")
display(plt)
savefig(plt, "reproduction_known_ssp.png")

# --- RMSE on the measurement points (compare with your 3 Jul number) ---
df   = CSV.read("ssnn_profiles_1224.csv", DataFrame)
pred = predict_amp(pm, df.range_m, df.depth_m)
println("Measurement-region RMSE = ", round(sqrt(mean((pred .- df.amp).^2)); sigdigits=4), " mVpp")
