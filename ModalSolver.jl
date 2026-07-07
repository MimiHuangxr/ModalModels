module ModalSolver

# Physics-aided normal-mode acoustic propagation model (MBNN).
#
# Optimization uses Julia's standard tooling: Optimisers.jl (the same Adam
# engine Flux uses under the hood, via `Flux.setup`/`Flux.Adam`). Gradient
# clipping uses `Optimisers.ClipNorm`, which — because the trainable parameters
# are a single flat vector — is exactly the global-L2-norm clip used before.
# To use Flux directly instead, replace `import Optimisers` with `import Flux`
# and `Optimisers.` with `Flux.` throughout `_train`.

using CSV
using DataFrames
using Random
using Statistics
import ForwardDiff
import Optimisers

export ModeSolver, fit!, train_mbnn
export predict_amp, predict_grid, learned_ssp, active_modes
export load_measurements, load_ssp, split_train_val

const KR_LO_FRAC = 0.02
const KR_HI_FRAC = 0.999

# -----------------------------------------------------------------------------
# Public solver type
# -----------------------------------------------------------------------------

"""
    ModeSolver(env=nothing; D, f, nmodes=6, nhidden=6,
               ssp=nothing, cmin=1445.0, cmax=1462.0,
               ngrid=200, rref=675.0)

Physics-aided normal-mode acoustic propagation model.

If `ssp` is provided, the solver uses the known Sound Speed Profile (SSP)
and learns only modal parameters `{A, B, kr}`.

If `ssp` is omitted, the solver embeds a Sound Speed Neural Network (SSNN)
and jointly learns a plausible SSP together with `{A, B, kr}`.

Depth is positive downward. Range is horizontal distance from source.
"""
mutable struct ModeSolver
    env::Any
    D::Float64
    f::Float64
    nmodes::Int
    nhidden::Int
    ssp::Any
    cmin::Float64
    cmax::Float64
    ngrid::Int
    rref::Float64
    branch::Symbol
    theta::Union{Nothing,Vector{Float64}}
    yscale::Float64
    history::DataFrame
    loss_weights::Dict{Symbol,Float64}
end

function ModeSolver(env=nothing; D, f, nmodes::Int=6, nhidden::Int=6,
                    ssp=nothing, cmin=1445.0, cmax=1462.0,
                    ngrid::Int=200, rref=675.0)
    nmodes > 0 || error("nmodes must be positive")
    nhidden > 0 || error("nhidden must be positive")
    D > 0 || error("D must be positive")
    f > 0 || error("f must be positive")
    branch = isnothing(ssp) ? :unknown_ssp : :known_ssp
    return ModeSolver(env, Float64(D), Float64(f), nmodes, nhidden, ssp,
                      Float64(cmin), Float64(cmax), ngrid, Float64(rref),
                      branch, nothing, 1.0, DataFrame(), Dict{Symbol,Float64}())
end

# -----------------------------------------------------------------------------
# CSV and data helpers
# -----------------------------------------------------------------------------

"""
    load_measurements(file; range_col=:range_m, depth_col=:depth_m, amp_col=:amp,
                      split_col=:split, keep_split=true)

Load measured acoustic amplitude data from a CSV file. Expected columns are
receiver range, receiver depth, and amplitude. A `split` column, if present, is
preserved for train/validation splitting.
"""
function load_measurements(file; range_col=:range_m, depth_col=:depth_m, amp_col=:amp,
                           split_col=:split, keep_split::Bool=true)
    df = CSV.read(file, DataFrame)
    for col in (range_col, depth_col, amp_col)
        hasproperty(df, col) || error("CSV is missing required column: $col")
    end
    out = (range_m=Float64.(df[:, range_col]), depth_m=Float64.(df[:, depth_col]),
           amp=Float64.(df[:, amp_col]))
    if keep_split && hasproperty(df, split_col)
        return merge(out, (split=String.(df[:, split_col]),))
    end
    return out
end

"""
    load_ssp(file; depth_col=:depth_m, c_col=:c_ms)

Load a Sound Speed Profile (SSP) CSV as `(depth_m, c_ms)`.
"""
function load_ssp(file; depth_col=:depth_m, c_col=:c_ms)
    df = CSV.read(file, DataFrame)
    hasproperty(df, depth_col) || error("CSV is missing required column: $depth_col")
    hasproperty(df, c_col) || error("CSV is missing required column: $c_col")
    return (Float64.(df[:, depth_col]), Float64.(df[:, c_col]))
end

"""
    split_train_val(data; val_fraction=0.0, seed=0)

Return `(train, val)` named tuples. If `data.split` exists, rows labelled
`train`/`val`/`validation` are used accordingly. Otherwise a random validation
split is used when `val_fraction > 0`; if not, validation equals training.
"""
function split_train_val(data; val_fraction=0.0, seed=0)
    _check_data(data)
    n = length(data.amp)
    if hasproperty(data, :split)
        labels = lowercase.(String.(data.split))
        train_idx = findall(==("train"), labels)
        val_idx = findall(x -> x == "val" || x == "validation", labels)
        isempty(train_idx) && error("split column exists, but no rows are labelled 'train'")
        isempty(val_idx) && (val_idx = train_idx)
        return _subset_data(data, train_idx), _subset_data(data, val_idx)
    end
    if val_fraction <= 0
        idx = collect(1:n)
        return _subset_data(data, idx), _subset_data(data, idx)
    end
    perm = randperm(MersenneTwister(seed), n)
    nval = max(1, round(Int, val_fraction * n))
    val_idx = perm[1:nval]
    train_idx = perm[(nval + 1):end]
    isempty(train_idx) && error("validation split leaves no training data")
    return _subset_data(data, train_idx), _subset_data(data, val_idx)
end

_subset_data(data, idx) =
    (range_m=data.range_m[idx], depth_m=data.depth_m[idx], amp=data.amp[idx])

function _check_data(data)
    for col in (:range_m, :depth_m, :amp)
        hasproperty(data, col) || error("data must contain `$col`")
    end
    length(data.range_m) == length(data.depth_m) == length(data.amp) ||
        error("data.range_m, data.depth_m, and data.amp must have the same length")
    all(data.range_m .> 0) || error("all ranges must be positive")
    all(data.depth_m .>= 0) || error("all depths must be non-negative")
    all(data.amp .>= 0) || error("all amplitudes must be non-negative")
    return true
end

# -----------------------------------------------------------------------------
# Shared math helpers
# -----------------------------------------------------------------------------

_sigmoid(x) = one(x) / (one(x) + exp(-x))
_logit(x) = log(x / (1 - x))
_relu(x) = max(zero(x), x)
_complex_abs_smooth(z) = sqrt(real(z)^2 + imag(z)^2 + 1e-12)
_rangefun(pm::ModeSolver, kr, r) = exp(im * kr * (r - pm.rref)) / sqrt(kr * r)

function _complex_l1(v)
    s = zero(real(v[1]))
    for x in v
        s += _complex_abs_smooth(x)
    end
    return s
end

function _cumtrapz(g, dz)
    out = similar(g)
    out[1] = zero(eltype(g))
    @inbounds for i in 2:length(g)
        out[i] = out[i - 1] + (g[i] + g[i - 1]) * (dz / 2)
    end
    return out
end

function _interp_depth(vals, dz, D, z)
    z <= 0 && return vals[1]
    z >= D && return vals[end]
    t = z / dz
    i = floor(Int, t) + 1
    w = t - (i - 1)
    return (1 - w) * vals[i] + w * vals[i + 1]
end

function _field_from_profiles(pm::ModeSolver, mode_profiles, kr, dz, r, z)
    acc = zero(eltype(mode_profiles[1]))
    for m in eachindex(kr)
        acc += _interp_depth(mode_profiles[m], dz, pm.D, z) * _rangefun(pm, kr[m], r)
    end
    return acc
end

# -----------------------------------------------------------------------------
# Sound speed profile helpers
# -----------------------------------------------------------------------------

function _ssp_function(ssp)
    if ssp isa Function
        return z -> Float64(ssp(z))
    elseif ssp isa Tuple && length(ssp) == 2
        return _linear_interp_function(Float64.(ssp[1]), Float64.(ssp[2]))
    elseif hasproperty(ssp, :depth_m) && hasproperty(ssp, :c_ms)
        return _linear_interp_function(Float64.(ssp.depth_m), Float64.(ssp.c_ms))
    else
        error("ssp must be a function z -> c, a tuple (depth_m, c_ms), or a named tuple with depth_m and c_ms")
    end
end

function _linear_interp_function(zvec, cvec)
    length(zvec) == length(cvec) || error("SSP depth and sound-speed vectors must have same length")
    length(zvec) >= 2 || error("SSP must contain at least two points")
    p = sortperm(zvec)
    z = zvec[p]
    c = cvec[p]
    return function (x)
        x <= z[1] && return c[1]
        x >= z[end] && return c[end]
        j = clamp(searchsortedlast(z, x), 1, length(z) - 1)
        t = (x - z[j]) / (z[j + 1] - z[j])
        return (1 - t) * c[j] + t * c[j + 1]
    end
end

function _ssnn_c(pm::ModeSolver, W1, b1, W2, b2, z)
    ζ = z / pm.D
    out = b2
    for j in 1:pm.nhidden
        out += W2[j] * _relu(W1[j] * ζ + b1[j])
    end
    return pm.cmin + (pm.cmax - pm.cmin) * _sigmoid(out)
end

# -----------------------------------------------------------------------------
# Unpacking, sound-speed grid, and mode construction (unified over both branches)
# -----------------------------------------------------------------------------

# Sound speed sampled on the depth grid: from the provided SSP (known branch),
# or from the embedded SSNN parameters (unknown branch).
function _cgrid(pm::ModeSolver, ssnn)
    if pm.branch == :known_ssp
        c = _ssp_function(pm.ssp)
    else
        W1, b1, W2, b2 = ssnn
        c = z -> _ssnn_c(pm, W1, b1, W2, b2, z)
    end
    return [c(z) for z in range(0.0, pm.D; length=pm.ngrid)]
end

# theta layout: [A_re; A_im; B_re; B_im; qkr] and, if unknown SSP, then
# [W1; b1; W2; b2] appended. Returns (A, B, kr, ssnn) with ssnn=nothing if known.
function _unpack(pm::ModeSolver, theta)
    n = pm.nmodes
    A = theta[1:n] .+ im .* theta[(n + 1):(2n)]
    B = theta[(2n + 1):(3n)] .+ im .* theta[(3n + 1):(4n)]
    qkr = theta[(4n + 1):(5n)]
    cmin_speed = pm.branch == :known_ssp ? minimum(_cgrid(pm, nothing)) : pm.cmin
    kmax = 2π * pm.f / cmin_speed
    kr = KR_LO_FRAC * kmax .+ (KR_HI_FRAC - KR_LO_FRAC) * kmax .* _sigmoid.(qkr)
    ssnn = nothing
    if pm.branch == :unknown_ssp
        nh = pm.nhidden
        o = 5n
        ssnn = (theta[(o + 1):(o + nh)], theta[(o + nh + 1):(o + 2nh)],
                theta[(o + 2nh + 1):(o + 3nh)], theta[o + 3nh + 1])
    end
    return A, B, kr, ssnn
end

function _build_modes(pm::ModeSolver, A, B, kr, cgrid)
    ω = 2π * pm.f
    dz = pm.D / (pm.ngrid - 1)
    T = eltype(kr)
    mode_profiles = Vector{Vector{Complex{T}}}(undef, pm.nmodes)
    for m in 1:pm.nmodes
        kz = @. sqrt(complex((ω / cgrid)^2 - kr[m]^2))
        kz_safe = kz .+ complex(1e-12)
        phase = _cumtrapz(kz, dz)
        mode_profiles[m] = @. A[m] * exp(im * phase) / sqrt(kz_safe) +
                              B[m] * exp(-im * phase) / sqrt(kz_safe)
    end
    return mode_profiles, dz
end

# -----------------------------------------------------------------------------
# Loss (unified): individual components, then the weighted sum
# -----------------------------------------------------------------------------

function _components(pm::ModeSolver, theta, data, idxs)
    A, B, kr, ssnn = _unpack(pm, theta)
    cgrid = _cgrid(pm, ssnn)
    mode_profiles, dz = _build_modes(pm, A, B, kr, cgrid)
    T = eltype(theta)

    L_amp = zero(T)
    L_log = zero(T)
    for ii in idxs
        pred = _complex_abs_smooth(_field_from_profiles(pm, mode_profiles, kr, dz,
                                                        data.range_m[ii], data.depth_m[ii]))
        truth = data.amp[ii]
        L_amp += (pred - truth)^2
        L_log += (log(pred + 1e-8) - log(truth + 1e-8))^2
    end
    L_amp /= length(idxs)
    L_log /= length(idxs)

    L_surface = zero(T)
    for m in 1:pm.nmodes
        L_surface += abs2(A[m] + B[m])
    end
    L_surface /= pm.nmodes

    L_smooth = zero(T)
    L_mono = zero(T)
    if pm.branch == :unknown_ssp
        for i in 2:(length(cgrid) - 1)
            L_smooth += (cgrid[i + 1] - 2 * cgrid[i] + cgrid[i - 1])^2
        end
        L_smooth /= length(cgrid)
        for i in 1:(length(cgrid) - 1)
            dc = cgrid[i + 1] - cgrid[i]
            L_mono += max(zero(dc), -dc)^2
        end
        L_mono /= length(cgrid)
    end

    return (amp=L_amp, log=L_log, surface=L_surface, smooth=L_smooth, mono=L_mono,
            A=_complex_l1(A), B=_complex_l1(B))
end

function _loss(pm::ModeSolver, theta, data, idxs, w)
    c = _components(pm, theta, data, idxs)
    L = w[:λ_amp] * c.amp + w[:λ_log] * c.log + w[:λ_surface] * c.surface +
        w[:α] * c.A + w[:β] * c.B
    pm.branch == :unknown_ssp && (L += w[:λ_smooth] * c.smooth + w[:λ_mono] * c.mono)
    return L
end

# -----------------------------------------------------------------------------
# Initialization (unified): physics-based kr guess; SSNN block if unknown SSP
# -----------------------------------------------------------------------------

function _init_theta(pm::ModeSolver, rng; c_init=(pm.cmin + pm.cmax) / 2)
    n = pm.nmodes
    if pm.branch == :known_ssp
        cgrid = _cgrid(pm, nothing)
        c_ref = mean(cgrid)
        cmin_speed = minimum(cgrid)
    else
        c_ref = c_init
        cmin_speed = pm.cmin
    end
    kref = 2π * pm.f / c_ref
    kmax = 2π * pm.f / cmin_speed
    kr_lo = KR_LO_FRAC * kmax
    kr_hi = KR_HI_FRAC * kmax
    qkr0 = [_logit(clamp((sqrt(max(kref^2 - ((m - 0.5) * π / pm.D)^2, 1e-8)) - kr_lo) /
                         (kr_hi - kr_lo), 1e-3, 1 - 1e-3)) for m in 1:n]

    theta = vcat(randn(rng, n), randn(rng, n), randn(rng, n), randn(rng, n), qkr0)
    if pm.branch == :unknown_ssp
        nh = pm.nhidden
        u_init = clamp((c_init - pm.cmin) / (pm.cmax - pm.cmin), 1e-3, 1 - 1e-3)
        theta = vcat(theta, 1.0 .* randn(rng, nh), 0.5 .* randn(rng, nh),
                     0.05 .* randn(rng, nh), [_logit(u_init)])
    end
    return theta
end

function _auto_balance_unknown!(pm::ModeSolver, theta, train_data, rng; sample_size=256)
    idxs = rand(rng, 1:length(train_data.amp), min(sample_size, length(train_data.amp)))
    c = _components(pm, theta, train_data, idxs)
    safe(x) = Float64(abs(x)) + 1e-12
    pm.loss_weights = Dict{Symbol,Float64}(
        :λ_amp => clamp(0.70 / safe(c.amp), 1e-6, 10.0),
        :λ_log => clamp(0.30 / safe(c.log), 1e-6, 10.0),
        :λ_surface => clamp(1e-3 / safe(c.surface), 1e-8, 1e-3),
        :λ_smooth => clamp(1e-3 / safe(c.smooth), 1e-8, 5e-3),
        :λ_mono => clamp(1e-3 / safe(c.mono), 1e-8, 5e-3),
        :α => clamp(1e-3 / safe(c.A), 1e-8, 1e-4),
        :β => clamp(1e-3 / safe(c.B), 1e-8, 1e-4),
    )
    return pm.loss_weights
end

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

"""
    fit!(pm, data; kwargs...)
    fit!(pm, csvfile; kwargs...)
    fit!(pm, tx, rxs, data; kwargs...)

Train the modal model on measured amplitudes. `tx` and `rxs` are accepted for
API compatibility, but this MBNN implementation reads the range/depth values
from `data`.

Common keyword arguments:
- `epochs`: training epochs per restart
- `batch_size`: mini-batch size
- `lr`: learning rate
- `restarts`: number of random restarts
- `val_fraction`: random validation split if data has no `split` column
- `seed`: reproducibility seed
- `log_every`: print interval
"""
function fit!(pm::ModeSolver, data; epochs=2000, batch_size=256, lr=nothing,
              restarts=10, val_fraction=0.0, seed=1000, log_every=250,
              verbose=true, auto_balance=true, kwargs...)
    _check_data(data)
    train_raw, val_raw = split_train_val(data; val_fraction=val_fraction, seed=seed)
    yscale = mean(train_raw.amp)
    yscale > 0 || error("mean training amplitude must be positive")
    pm.yscale = yscale
    train_data = (range_m=train_raw.range_m, depth_m=train_raw.depth_m, amp=train_raw.amp ./ yscale)
    val_data = (range_m=val_raw.range_m, depth_m=val_raw.depth_m, amp=val_raw.amp ./ yscale)

    if pm.branch == :known_ssp
        return _fit_known_ssp!(pm, train_data, val_data; epochs, batch_size,
            lr=isnothing(lr) ? 3e-2 : lr, restarts, seed, log_every, verbose, kwargs...)
    else
        return _fit_unknown_ssp!(pm, train_data, val_data; epochs, batch_size,
            lr=isnothing(lr) ? 4e-2 : lr, restarts, seed, log_every, verbose, auto_balance, kwargs...)
    end
end

fit!(pm::ModeSolver, csvfile::AbstractString; kwargs...) =
    fit!(pm, load_measurements(csvfile); kwargs...)
fit!(pm::ModeSolver, tx, rxs, data; kwargs...) = fit!(pm, data; kwargs...)
fit!(pm::ModeSolver, tx, rxs, csvfile::AbstractString; kwargs...) =
    fit!(pm, load_measurements(csvfile); kwargs...)

# Adam training via Optimisers.jl. ClipNorm(100) reproduces the previous
# global-L2-norm gradient clip (theta is a single parameter leaf), and
# Optimisers.Adam is the standard Adam update (β=(0.9,0.999), ϵ=1e-8).
function _train(loss, theta0; epochs, lr, batch_loss, batch_size, ntrain,
                rng, log_every, verbose, val_loss)
    theta = copy(theta0)
    rule = Optimisers.OptimiserChain(Optimisers.ClipNorm(100.0), Optimisers.Adam(lr))
    state = Optimisers.setup(rule, theta)

    best_theta = copy(theta)
    best_val = Inf
    best_epoch = 0
    epoch_hist = Int[]
    train_hist = Float64[]
    val_hist = Float64[]

    for epoch in 1:epochs
        idxs = rand(rng, 1:ntrain, min(batch_size, ntrain))
        g = ForwardDiff.gradient(t -> batch_loss(t, idxs), theta)
        state, theta = Optimisers.update!(state, theta, g)

        if epoch == 1 || epoch % log_every == 0 || epoch == epochs
            tl = Float64(loss(theta))
            vl = Float64(val_loss(theta))
            push!(epoch_hist, epoch)
            push!(train_hist, tl)
            push!(val_hist, vl)
            if vl < best_val
                best_val = vl
                best_theta = copy(theta)
                best_epoch = epoch
            end
            verbose && println("epoch $epoch | train=$(round(tl; sigdigits=5)) | val=$(round(vl; sigdigits=5))")
        end
    end
    return best_theta, best_val, best_epoch, epoch_hist, train_hist, val_hist
end

function _fit_known_ssp!(pm::ModeSolver, train_data, val_data; epochs, batch_size, lr,
                         restarts, seed, log_every, verbose,
                         λ_amp=1.0, λ_log=0.0, λ_surface=1e-4, α=1e-6, β=1e-6)
    weights = Dict{Symbol,Float64}(:λ_amp => λ_amp, :λ_log => λ_log,
                                   :λ_surface => λ_surface, :α => α, :β => β)
    pm.loss_weights = weights
    best = (theta=nothing, val=Inf, restart=0, epoch=0, hist=DataFrame())

    for restart in 1:restarts
        rng = MersenneTwister(seed + restart)
        theta0 = _init_theta(pm, rng)
        verbose && println("\nKnown-SSP restart $restart / $restarts")

        train_loss = theta -> _loss(pm, theta, train_data, eachindex(train_data.amp), weights)
        val_loss = theta -> _loss(pm, theta, val_data, eachindex(val_data.amp), weights)
        batch_loss = (theta, idxs) -> _loss(pm, theta, train_data, idxs, weights)

        theta, val, be, eh, th, vh = _train(train_loss, theta0; epochs, lr, batch_loss,
            batch_size, ntrain=length(train_data.amp), rng, log_every, verbose, val_loss)
        if val < best.val
            best = (theta=copy(theta), val=val, restart=restart, epoch=be,
                    hist=DataFrame(epoch=eh, train_loss=th, val_loss=vh))
        end
    end

    pm.theta = best.theta
    pm.history = best.hist
    verbose && println("\nBest known-SSP run: restart=$(best.restart) epoch=$(best.epoch) val=$(best.val)")
    return pm
end

function _fit_unknown_ssp!(pm::ModeSolver, train_data, val_data; epochs, batch_size, lr,
                           restarts, seed, log_every, verbose, auto_balance=true)
    best = (theta=nothing, val=Inf, restart=0, epoch=0, hist=DataFrame(), weights=Dict{Symbol,Float64}())

    for restart in 1:restarts
        rng = MersenneTwister(seed + restart)
        theta0 = _init_theta(pm, rng)

        if auto_balance || isempty(pm.loss_weights)
            _auto_balance_unknown!(pm, theta0, train_data, rng)
        else
            defaults = Dict{Symbol,Float64}(:λ_amp => 0.70, :λ_log => 0.30,
                :λ_surface => 1e-4, :λ_smooth => 8e-4, :λ_mono => 8e-4, :α => 1e-6, :β => 1e-6)
            merge!(defaults, pm.loss_weights)
            pm.loss_weights = defaults
        end
        weights = pm.loss_weights

        verbose && println("\nUnknown-SSP restart $restart / $restarts")
        verbose && println("loss weights = ", weights)

        train_loss = theta -> _loss(pm, theta, train_data, eachindex(train_data.amp), weights)
        val_loss = theta -> _loss(pm, theta, val_data, eachindex(val_data.amp), weights)
        batch_loss = (theta, idxs) -> _loss(pm, theta, train_data, idxs, weights)

        theta, val, be, eh, th, vh = _train(train_loss, theta0; epochs, lr, batch_loss,
            batch_size, ntrain=length(train_data.amp), rng, log_every, verbose, val_loss)
        if val < best.val
            best = (theta=copy(theta), val=val, restart=restart, epoch=be,
                    hist=DataFrame(epoch=eh, train_loss=th, val_loss=vh), weights=copy(weights))
        end
    end

    pm.theta = best.theta
    pm.history = best.hist
    pm.loss_weights = best.weights
    verbose && println("\nBest unknown-SSP run: restart=$(best.restart) epoch=$(best.epoch) val=$(best.val)")
    return pm
end

"""
    train_mbnn(range_m, depth_m, amp; kwargs...)

Convenience wrapper matching the README style. Returns a fitted `ModeSolver`.
"""
function train_mbnn(range_m, depth_m, amp; env=nothing, D, f, nmodes::Int=6, nhidden::Int=6,
                    ssp=nothing, cmin=1445.0, cmax=1462.0, ngrid::Int=200, rref=675.0, kwargs...)
    pm = ModeSolver(env; D, f, nmodes, nhidden, ssp, cmin, cmax, ngrid, rref)
    fit!(pm, (range_m=Float64.(range_m), depth_m=Float64.(depth_m), amp=Float64.(amp)); kwargs...)
    return pm
end

# -----------------------------------------------------------------------------
# Prediction and outputs
# -----------------------------------------------------------------------------

function _require_fitted(pm::ModeSolver)
    isnothing(pm.theta) && error("model has not been fitted yet. Call fit!(pm, data) first.")
    return true
end

"""
    predict_amp(pm, ranges, depths)

Predict linear pressure amplitude at matching `(range, depth)` receiver points.
"""
function predict_amp(pm::ModeSolver, ranges, depths)
    _require_fitted(pm)
    rs = Float64.(collect(ranges))
    zs = Float64.(collect(depths))
    length(rs) == length(zs) || error("ranges and depths must have the same length")
    A, B, kr, ssnn = _unpack(pm, pm.theta)
    mode_profiles, dz = _build_modes(pm, A, B, kr, _cgrid(pm, ssnn))
    out = Vector{Float64}(undef, length(rs))
    for i in eachindex(rs)
        out[i] = Float64(_complex_abs_smooth(_field_from_profiles(pm, mode_profiles, kr, dz, rs[i], zs[i]))) * pm.yscale
    end
    return out
end

"""
    predict_grid(pm, ranges, depths)

Predict a full range-depth amplitude grid. Rows correspond to `depths`; columns
correspond to `ranges`.
"""
function predict_grid(pm::ModeSolver, ranges, depths)
    _require_fitted(pm)
    rs = Float64.(collect(ranges))
    zs = Float64.(collect(depths))
    A, B, kr, ssnn = _unpack(pm, pm.theta)
    mode_profiles, dz = _build_modes(pm, A, B, kr, _cgrid(pm, ssnn))
    out = Matrix{Float64}(undef, length(zs), length(rs))
    for (iz, z) in enumerate(zs)
        mode_vals = [_interp_depth(mode_profiles[m], dz, pm.D, z) for m in eachindex(kr)]
        for (ir, r) in enumerate(rs)
            p = zero(eltype(mode_profiles[1]))
            for m in eachindex(kr)
                p += mode_vals[m] * _rangefun(pm, kr[m], r)
            end
            out[iz, ir] = Float64(_complex_abs_smooth(p)) * pm.yscale
        end
    end
    return out
end

"""
    learned_ssp(pm)

Return a function `c(z)` for the learned SSP if SSP was unknown. If SSP was
known, return the provided SSP interpolation function.
"""
function learned_ssp(pm::ModeSolver)
    pm.branch == :known_ssp && return _ssp_function(pm.ssp)
    _require_fitted(pm)
    _, _, _, ssnn = _unpack(pm, pm.theta)
    W1, b1, W2, b2 = ssnn
    return z -> Float64(_ssnn_c(pm, W1, b1, W2, b2, z))
end

"""
    active_modes(pm)

Return learned modal parameters ranked by `|A| + |B|`.
"""
function active_modes(pm::ModeSolver)
    _require_fitted(pm)
    A, B, kr, _ = _unpack(pm, pm.theta)
    df = DataFrame(
        mode=collect(1:pm.nmodes),
        amplitude=Float64.([_complex_abs_smooth(A[i]) + _complex_abs_smooth(B[i]) for i in 1:pm.nmodes]),
        kr=Float64.(kr),
        A_re=Float64.(real.(A)), A_im=Float64.(imag.(A)),
        B_re=Float64.(real.(B)), B_im=Float64.(imag.(B)),
    )
    sort!(df, :amplitude, rev=true)
    return df
end

end # module ModalSolver
