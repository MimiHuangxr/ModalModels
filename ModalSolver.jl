module ModalSolver

using CSV
using DataFrames
using Random
using Statistics
using LinearAlgebra
import ForwardDiff

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
    return ModeSolver(
        env,
        Float64(D),
        Float64(f),
        nmodes,
        nhidden,
        ssp,
        Float64(cmin),
        Float64(cmax),
        ngrid,
        Float64(rref),
        branch,
        nothing,
        1.0,
        DataFrame(),
        Dict{Symbol,Float64}(),
    )
end

# -----------------------------------------------------------------------------
# CSV and data helpers
# -----------------------------------------------------------------------------

"""
    load_measurements(file; range_col=:range_m, depth_col=:depth_m, amp_col=:amp,
                      split_col=:split, keep_split=true)

Load measured acoustic amplitude data from a CSV file.
Expected columns are receiver range, receiver depth, and amplitude.
If a `split` column exists, it is preserved for train/validation splitting.
"""
function load_measurements(file; range_col=:range_m, depth_col=:depth_m, amp_col=:amp,
                           split_col=:split, keep_split::Bool=true)
    df = CSV.read(file, DataFrame)
    for col in (range_col, depth_col, amp_col)
        hasproperty(df, col) || error("CSV is missing required column: $col")
    end

    out = (
        range_m = Float64.(df[:, range_col]),
        depth_m = Float64.(df[:, depth_col]),
        amp = Float64.(df[:, amp_col]),
    )

    if keep_split && hasproperty(df, split_col)
        return merge(out, (split = String.(df[:, split_col]),))
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

Return `(train, val)` named tuples.
If `data.split` exists, rows labelled `train` go to training and rows labelled
`val` or `validation` go to validation. Otherwise, a random validation split is
used when `val_fraction > 0`; if not, validation equals training.
"""
function split_train_val(data; val_fraction=0.0, seed=0)
    _check_data(data)
    n = length(data.amp)

    if hasproperty(data, :split)
        labels = lowercase.(String.(data.split))
        train_idx = findall(x -> x == "train", labels)
        val_idx = findall(x -> x == "val" || x == "validation", labels)
        isempty(train_idx) && error("split column exists, but no rows are labelled 'train'")
        isempty(val_idx) && (val_idx = train_idx)
        return _subset_data(data, train_idx), _subset_data(data, val_idx)
    end

    if val_fraction <= 0
        idx = collect(1:n)
        return _subset_data(data, idx), _subset_data(data, idx)
    end

    rng = MersenneTwister(seed)
    perm = randperm(rng, n)
    nval = max(1, round(Int, val_fraction * n))
    val_idx = perm[1:nval]
    train_idx = perm[(nval + 1):end]
    isempty(train_idx) && error("validation split leaves no training data")
    return _subset_data(data, train_idx), _subset_data(data, val_idx)
end

function _subset_data(data, idx)
    return (
        range_m = data.range_m[idx],
        depth_m = data.depth_m[idx],
        amp = data.amp[idx],
    )
end

function _check_data(data)
    hasproperty(data, :range_m) || error("data must contain `range_m`")
    hasproperty(data, :depth_m) || error("data must contain `depth_m`")
    hasproperty(data, :amp) || error("data must contain `amp`")
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
        zvec = Float64.(ssp[1])
        cvec = Float64.(ssp[2])
        return _linear_interp_function(zvec, cvec)
    elseif hasproperty(ssp, :depth_m) && hasproperty(ssp, :c_ms)
        zvec = Float64.(ssp.depth_m)
        cvec = Float64.(ssp.c_ms)
        return _linear_interp_function(zvec, cvec)
    else
        error("ssp must be either a function z -> c, a tuple (depth_m, c_ms), or a named tuple with depth_m and c_ms")
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
        j = searchsortedlast(z, x)
        j = clamp(j, 1, length(z) - 1)
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
# Parameter unpacking
# -----------------------------------------------------------------------------

function _unpack_known(pm::ModeSolver, theta)
    n = pm.nmodes
    A = theta[1:n] .+ im .* theta[(n + 1):(2 * n)]
    B = theta[(2 * n + 1):(3 * n)] .+ im .* theta[(3 * n + 1):(4 * n)]
    qkr = theta[(4 * n + 1):(5 * n)]

    c_fn = _ssp_function(pm.ssp)
    cgrid = [c_fn(z) for z in range(0.0, pm.D; length=pm.ngrid)]
    kmax = 2π * pm.f / minimum(cgrid)
    kr_lo = KR_LO_FRAC * kmax
    kr_hi = KR_HI_FRAC * kmax
    kr = kr_lo .+ (kr_hi - kr_lo) .* _sigmoid.(qkr)
    return A, B, kr
end

function _unpack_unknown(pm::ModeSolver, theta)
    n = pm.nmodes
    nh = pm.nhidden
    A = theta[1:n] .+ im .* theta[(n + 1):(2 * n)]
    B = theta[(2 * n + 1):(3 * n)] .+ im .* theta[(3 * n + 1):(4 * n)]
    qkr = theta[(4 * n + 1):(5 * n)]

    kmax = 2π * pm.f / pm.cmin
    kr_lo = KR_LO_FRAC * kmax
    kr_hi = KR_HI_FRAC * kmax
    kr = kr_lo .+ (kr_hi - kr_lo) .* _sigmoid.(qkr)

    o = 5 * n
    W1 = theta[(o + 1):(o + nh)]
    b1 = theta[(o + nh + 1):(o + 2 * nh)]
    W2 = theta[(o + 2 * nh + 1):(o + 3 * nh)]
    b2 = theta[o + 3 * nh + 1]

    return A, B, kr, W1, b1, W2, b2
end

# -----------------------------------------------------------------------------
# Mode construction
# -----------------------------------------------------------------------------

function _build_known_modes(pm::ModeSolver, theta)
    A, B, kr = _unpack_known(pm, theta)
    c_fn = _ssp_function(pm.ssp)
    ω = 2π * pm.f
    dz = pm.D / (pm.ngrid - 1)
    zgrid = range(0.0, pm.D; length=pm.ngrid)
    cgrid = [c_fn(z) for z in zgrid]

    T = eltype(kr)
    mode_profiles = Vector{Vector{Complex{T}}}(undef, pm.nmodes)

    for m in 1:pm.nmodes
        kz = @. sqrt(complex((ω / cgrid)^2 - kr[m]^2))
        kz_safe = kz .+ complex(1e-12)
        phase = _cumtrapz(kz, dz)
        mode_profiles[m] = @. A[m] * exp(im * phase) / sqrt(kz_safe) +
                              B[m] * exp(-im * phase) / sqrt(kz_safe)
    end

    return mode_profiles, dz, kr
end

function _build_unknown_modes(pm::ModeSolver, theta)
    A, B, kr, W1, b1, W2, b2 = _unpack_unknown(pm, theta)
    ω = 2π * pm.f
    dz = pm.D / (pm.ngrid - 1)
    zgrid = range(0.0, pm.D; length=pm.ngrid)
    cgrid = [_ssnn_c(pm, W1, b1, W2, b2, z) for z in zgrid]

    T = eltype(kr)
    mode_profiles = Vector{Vector{Complex{T}}}(undef, pm.nmodes)

    for m in 1:pm.nmodes
        kz = @. sqrt(complex((ω / cgrid)^2 - kr[m]^2))
        kz_safe = kz .+ complex(1e-12)
        phase = _cumtrapz(kz, dz)
        mode_profiles[m] = @. A[m] * exp(im * phase) / sqrt(kz_safe) +
                              B[m] * exp(-im * phase) / sqrt(kz_safe)
    end

    return mode_profiles, dz, kr, cgrid
end

# -----------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------

function _known_loss(pm::ModeSolver, theta, data, idxs;
                     λ_amp=1.0, λ_log=0.0, λ_surface=1e-4,
                     α=1e-6, β=1e-6, eps_amp=1e-8)
    A, B, kr = _unpack_known(pm, theta)
    mode_profiles, dz, kr2 = _build_known_modes(pm, theta)

    L_amp = zero(eltype(theta))
    L_log = zero(eltype(theta))

    for ii in idxs
        p = _field_from_profiles(pm, mode_profiles, kr2, dz, data.range_m[ii], data.depth_m[ii])
        pred = _complex_abs_smooth(p)
        truth = data.amp[ii]
        L_amp += (pred - truth)^2
        L_log += (log(pred + eps_amp) - log(truth + eps_amp))^2
    end

    L_amp /= length(idxs)
    L_log /= length(idxs)

    L_surface = zero(eltype(theta))
    for m in 1:pm.nmodes
        L_surface += real(A[m] + B[m])^2 + imag(A[m] + B[m])^2
    end
    L_surface /= pm.nmodes

    return λ_amp * L_amp + λ_log * L_log +
           λ_surface * L_surface + α * _complex_l1(A) + β * _complex_l1(B)
end

function _unknown_components(pm::ModeSolver, theta, data, idxs)
    A, B, kr, W1, b1, W2, b2 = _unpack_unknown(pm, theta)
    mode_profiles, dz, kr2, cgrid = _build_unknown_modes(pm, theta)

    L_amp = zero(eltype(theta))
    L_log = zero(eltype(theta))

    for ii in idxs
        p = _field_from_profiles(pm, mode_profiles, kr2, dz, data.range_m[ii], data.depth_m[ii])
        pred = _complex_abs_smooth(p)
        truth = data.amp[ii]
        L_amp += (pred - truth)^2
        L_log += (log(pred + 1e-8) - log(truth + 1e-8))^2
    end

    L_amp /= length(idxs)
    L_log /= length(idxs)

    L_surface = zero(eltype(theta))
    for m in 1:pm.nmodes
        L_surface += real(A[m] + B[m])^2 + imag(A[m] + B[m])^2
    end
    L_surface /= pm.nmodes

    L_smooth = zero(eltype(theta))
    for i in 2:(length(cgrid) - 1)
        L_smooth += (cgrid[i + 1] - 2 * cgrid[i] + cgrid[i - 1])^2
    end
    L_smooth /= length(cgrid)

    L_mono = zero(eltype(theta))
    for i in 1:(length(cgrid) - 1)
        dc = cgrid[i + 1] - cgrid[i]
        L_mono += max(zero(dc), -dc)^2
    end
    L_mono /= length(cgrid)

    L_A = _complex_l1(A)
    L_B = _complex_l1(B)

    return L_amp, L_log, L_surface, L_smooth, L_mono, L_A, L_B
end

function _unknown_loss(pm::ModeSolver, theta, data, idxs; weights=pm.loss_weights)
    L_amp, L_log, L_surface, L_smooth, L_mono, L_A, L_B =
        _unknown_components(pm, theta, data, idxs)

    return weights[:λ_amp] * L_amp +
           weights[:λ_log] * L_log +
           weights[:λ_surface] * L_surface +
           weights[:λ_smooth] * L_smooth +
           weights[:λ_mono] * L_mono +
           weights[:α] * L_A +
           weights[:β] * L_B
end

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------

function _init_known_theta(pm::ModeSolver, rng)
    n = pm.nmodes
    c_fn = _ssp_function(pm.ssp)
    cgrid = [c_fn(z) for z in range(0.0, pm.D; length=pm.ngrid)]
    c_ref = mean(cgrid)
    kref = 2π * pm.f / c_ref
    kmax = 2π * pm.f / minimum(cgrid)
    kr_lo = KR_LO_FRAC * kmax
    kr_hi = KR_HI_FRAC * kmax

    qkr0 = Float64[]
    for m in 1:n
        kz_guess = (m - 0.5) * π / pm.D
        kr_guess = sqrt(max(kref^2 - kz_guess^2, 1e-8))
        u = clamp((kr_guess - kr_lo) / (kr_hi - kr_lo), 1e-3, 1 - 1e-3)
        push!(qkr0, _logit(u))
    end

    return vcat(
        1.0 .* randn(rng, n),
        1.0 .* randn(rng, n),
        1.0 .* randn(rng, n),
        1.0 .* randn(rng, n),
        qkr0,
    )
end

function _init_unknown_theta(pm::ModeSolver, rng; c_init=(pm.cmin + pm.cmax) / 2)
    n = pm.nmodes
    nh = pm.nhidden
    kref = 2π * pm.f / c_init
    kmax = 2π * pm.f / pm.cmin
    kr_lo = KR_LO_FRAC * kmax
    kr_hi = KR_HI_FRAC * kmax

    qkr0 = Float64[]
    for m in 1:n
        kz_guess = (m - 0.5) * π / pm.D
        kr_guess = sqrt(max(kref^2 - kz_guess^2, 1e-8))
        u = clamp((kr_guess - kr_lo) / (kr_hi - kr_lo), 1e-3, 1 - 1e-3)
        push!(qkr0, _logit(u))
    end

    u_init = clamp((c_init - pm.cmin) / (pm.cmax - pm.cmin), 1e-3, 1 - 1e-3)
    W1_0 = 1.0 .* randn(rng, nh)
    b1_0 = 0.5 .* randn(rng, nh)
    W2_0 = 0.05 .* randn(rng, nh)
    b2_0 = _logit(u_init)

    return vcat(
        1.0 .* randn(rng, n),
        1.0 .* randn(rng, n),
        1.0 .* randn(rng, n),
        1.0 .* randn(rng, n),
        qkr0,
        W1_0,
        b1_0,
        W2_0,
        [b2_0],
    )
end

function _auto_balance_unknown!(pm::ModeSolver, theta, train_data, rng; sample_size=256)
    idxs = rand(rng, 1:length(train_data.amp), min(sample_size, length(train_data.amp)))
    L_amp, L_log, L_surface, L_smooth, L_mono, L_A, L_B =
        _unknown_components(pm, theta, train_data, idxs)
    safe(x) = Float64(abs(x)) + 1e-12

    pm.loss_weights = Dict{Symbol,Float64}(
        :λ_amp => clamp(0.70 / safe(L_amp), 1e-6, 10.0),
        :λ_log => clamp(0.30 / safe(L_log), 1e-6, 10.0),
        :λ_surface => clamp(1e-3 / safe(L_surface), 1e-8, 1e-3),
        :λ_smooth => clamp(1e-3 / safe(L_smooth), 1e-8, 5e-3),
        :λ_mono => clamp(1e-3 / safe(L_mono), 1e-8, 5e-3),
        :α => clamp(1e-3 / safe(L_A), 1e-8, 1e-4),
        :β => clamp(1e-3 / safe(L_B), 1e-8, 1e-4),
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

Train the modal model on measured amplitudes.
`tx` and `rxs` are accepted for compatibility, but this MBNN implementation uses
the range/depth values inside `data`.

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
              restarts=1, val_fraction=0.0, seed=1000, log_every=250,
              verbose=true, auto_balance=true, kwargs...)
    _check_data(data)

    train_raw, val_raw = split_train_val(data; val_fraction=val_fraction, seed=seed)
    yscale = mean(train_raw.amp)
    yscale > 0 || error("mean training amplitude must be positive")
    pm.yscale = yscale

    train_data = (range_m=train_raw.range_m, depth_m=train_raw.depth_m, amp=train_raw.amp ./ yscale)
    val_data = (range_m=val_raw.range_m, depth_m=val_raw.depth_m, amp=val_raw.amp ./ yscale)

    if pm.branch == :known_ssp
        return _fit_known_ssp!(pm, train_data, val_data;
            epochs=epochs,
            batch_size=batch_size,
            lr=isnothing(lr) ? 3e-2 : lr,
            restarts=restarts,
            seed=seed,
            log_every=log_every,
            verbose=verbose,
            kwargs...,
        )
    else
        return _fit_unknown_ssp!(pm, train_data, val_data;
            epochs=epochs,
            batch_size=batch_size,
            lr=isnothing(lr) ? 4e-2 : lr,
            restarts=restarts,
            seed=seed,
            log_every=log_every,
            verbose=verbose,
            auto_balance=auto_balance,
            kwargs...,
        )
    end
end

function fit!(pm::ModeSolver, csvfile::AbstractString; kwargs...)
    data = load_measurements(csvfile)
    return fit!(pm, data; kwargs...)
end

function fit!(pm::ModeSolver, tx, rxs, data; kwargs...)
    return fit!(pm, data; kwargs...)
end

function fit!(pm::ModeSolver, tx, rxs, csvfile::AbstractString; kwargs...)
    data = load_measurements(csvfile)
    return fit!(pm, data; kwargs...)
end

function _adam_train(loss, theta0; epochs, lr, batch_loss, batch_size, ntrain,
                     rng, log_every, verbose, val_loss)
    theta = copy(theta0)
    m = zeros(length(theta))
    v = zeros(length(theta))
    β1 = 0.9
    β2 = 0.999
    eps_adam = 1e-8

    best_theta = copy(theta)
    best_val = Inf
    best_epoch = 0

    epoch_hist = Int[]
    train_hist = Float64[]
    val_hist = Float64[]

    for epoch in 1:epochs
        idxs = rand(rng, 1:ntrain, min(batch_size, ntrain))
        g = ForwardDiff.gradient(t -> batch_loss(t, idxs), theta)

        gnorm = norm(g)
        if isfinite(gnorm) && gnorm > 100.0
            g .*= 100.0 / gnorm
        end

        @. m = β1 * m + (1 - β1) * g
        @. v = β2 * v + (1 - β2) * g^2

        mh = m ./ (1 - β1^epoch)
        vh = v ./ (1 - β2^epoch)
        @. theta = theta - lr * mh / (sqrt(vh) + eps_adam)

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

function _fit_known_ssp!(pm::ModeSolver, train_data, val_data;
                         epochs, batch_size, lr, restarts, seed, log_every, verbose,
                         λ_amp=1.0, λ_log=0.0, λ_surface=1e-4, α=1e-6, β=1e-6)
    pm.loss_weights = Dict{Symbol,Float64}(
        :λ_amp => λ_amp,
        :λ_log => λ_log,
        :λ_surface => λ_surface,
        :α => α,
        :β => β,
    )

    global_best_theta = nothing
    global_best_val = Inf
    global_best_restart = 0
    global_best_epoch = 0
    best_history = DataFrame()

    for restart in 1:restarts
        rng = MersenneTwister(seed + restart)
        theta0 = _init_known_theta(pm, rng)
        verbose && println("\nKnown-SSP restart $restart / $restarts")

        full_train = theta -> _known_loss(pm, theta, train_data, eachindex(train_data.amp);
            λ_amp=λ_amp, λ_log=λ_log, λ_surface=λ_surface, α=α, β=β)
        full_val = theta -> _known_loss(pm, theta, val_data, eachindex(val_data.amp);
            λ_amp=λ_amp, λ_log=λ_log, λ_surface=λ_surface, α=α, β=β)
        batch_loss = (theta, idxs) -> _known_loss(pm, theta, train_data, idxs;
            λ_amp=λ_amp, λ_log=λ_log, λ_surface=λ_surface, α=α, β=β)

        theta, val, best_epoch, eh, th, vh = _adam_train(
            full_train,
            theta0;
            epochs=epochs,
            lr=lr,
            batch_loss=batch_loss,
            batch_size=batch_size,
            ntrain=length(train_data.amp),
            rng=rng,
            log_every=log_every,
            verbose=verbose,
            val_loss=full_val,
        )

        if val < global_best_val
            global_best_theta = copy(theta)
            global_best_val = val
            global_best_restart = restart
            global_best_epoch = best_epoch
            best_history = DataFrame(epoch=eh, train_loss=th, val_loss=vh)
        end
    end

    pm.theta = global_best_theta
    pm.history = best_history
    verbose && println("\nBest known-SSP run: restart=$global_best_restart epoch=$global_best_epoch val=$global_best_val")
    return pm
end

function _fit_unknown_ssp!(pm::ModeSolver, train_data, val_data;
                           epochs, batch_size, lr, restarts, seed, log_every, verbose,
                           auto_balance=true)
    global_best_theta = nothing
    global_best_val = Inf
    global_best_restart = 0
    global_best_epoch = 0
    best_history = DataFrame()
    best_weights = Dict{Symbol,Float64}()

    for restart in 1:restarts
        rng = MersenneTwister(seed + restart)
        theta0 = _init_unknown_theta(pm, rng)

        if auto_balance || isempty(pm.loss_weights)
            _auto_balance_unknown!(pm, theta0, train_data, rng)
        else
            defaults = Dict{Symbol,Float64}(
                :λ_amp => 0.70,
                :λ_log => 0.30,
                :λ_surface => 1e-4,
                :λ_smooth => 8e-4,
                :λ_mono => 8e-4,
                :α => 1e-6,
                :β => 1e-6,
            )
            merge!(defaults, pm.loss_weights)
            pm.loss_weights = defaults
        end

        verbose && println("\nUnknown-SSP restart $restart / $restarts")
        verbose && println("loss weights = ", pm.loss_weights)

        full_train = theta -> _unknown_loss(pm, theta, train_data, eachindex(train_data.amp))
        full_val = theta -> _unknown_loss(pm, theta, val_data, eachindex(val_data.amp))
        batch_loss = (theta, idxs) -> _unknown_loss(pm, theta, train_data, idxs)

        theta, val, best_epoch, eh, th, vh = _adam_train(
            full_train,
            theta0;
            epochs=epochs,
            lr=lr,
            batch_loss=batch_loss,
            batch_size=batch_size,
            ntrain=length(train_data.amp),
            rng=rng,
            log_every=log_every,
            verbose=verbose,
            val_loss=full_val,
        )

        if val < global_best_val
            global_best_theta = copy(theta)
            global_best_val = val
            global_best_restart = restart
            global_best_epoch = best_epoch
            best_history = DataFrame(epoch=eh, train_loss=th, val_loss=vh)
            best_weights = copy(pm.loss_weights)
        end
    end

    pm.theta = global_best_theta
    pm.history = best_history
    pm.loss_weights = best_weights
    verbose && println("\nBest unknown-SSP run: restart=$global_best_restart epoch=$global_best_epoch val=$global_best_val")
    return pm
end

"""
    train_mbnn(range_m, depth_m, amp; kwargs...)

Convenience wrapper matching the README style. Returns a fitted `ModeSolver`.
"""
function train_mbnn(range_m, depth_m, amp; env=nothing, D, f, nmodes::Int=6, nhidden::Int=6,
                    ssp=nothing, cmin=1445.0, cmax=1462.0, ngrid::Int=200,
                    rref=675.0, kwargs...)
    pm = ModeSolver(
        env;
        D=D,
        f=f,
        nmodes=nmodes,
        nhidden=nhidden,
        ssp=ssp,
        cmin=cmin,
        cmax=cmax,
        ngrid=ngrid,
        rref=rref,
    )
    data = (range_m=Float64.(range_m), depth_m=Float64.(depth_m), amp=Float64.(amp))
    fit!(pm, data; kwargs...)
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

    if pm.branch == :known_ssp
        mode_profiles, dz, kr = _build_known_modes(pm, pm.theta)
    else
        mode_profiles, dz, kr, _ = _build_unknown_modes(pm, pm.theta)
    end

    out = Vector{Float64}(undef, length(rs))
    for i in eachindex(rs)
        p = _field_from_profiles(pm, mode_profiles, kr, dz, rs[i], zs[i])
        out[i] = Float64(_complex_abs_smooth(p)) * pm.yscale
    end
    return out
end

"""
    predict_grid(pm, ranges, depths)

Predict a full range-depth amplitude grid.
Rows correspond to `depths`; columns correspond to `ranges`.
"""
function predict_grid(pm::ModeSolver, ranges, depths)
    _require_fitted(pm)
    rs = Float64.(collect(ranges))
    zs = Float64.(collect(depths))

    if pm.branch == :known_ssp
        mode_profiles, dz, kr = _build_known_modes(pm, pm.theta)
    else
        mode_profiles, dz, kr, _ = _build_unknown_modes(pm, pm.theta)
    end

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

Return a function `c(z)` for the learned SSP if SSP was unknown.
If SSP was known, return the provided SSP interpolation function.
"""
function learned_ssp(pm::ModeSolver)
    if pm.branch == :known_ssp
        return _ssp_function(pm.ssp)
    end

    _require_fitted(pm)
    _, _, _, W1, b1, W2, b2 = _unpack_unknown(pm, pm.theta)
    return z -> Float64(_ssnn_c(pm, W1, b1, W2, b2, z))
end

"""
    active_modes(pm)

Return learned modal parameters ranked by `|A| + |B|`.
"""
function active_modes(pm::ModeSolver)
    _require_fitted(pm)

    if pm.branch == :known_ssp
        A, B, kr = _unpack_known(pm, pm.theta)
    else
        A, B, kr, _, _, _, _ = _unpack_unknown(pm, pm.theta)
    end

    df = DataFrame(
        mode = collect(1:pm.nmodes),
        amplitude = Float64.([_complex_abs_smooth(A[i]) + _complex_abs_smooth(B[i]) for i in 1:pm.nmodes]),
        kr = Float64.(kr),
        A_re = Float64.(real.(A)),
        A_im = Float64.(imag.(A)),
        B_re = Float64.(real.(B)),
        B_im = Float64.(imag.(B)),
    )
    sort!(df, :amplitude, rev=true)
    return df
end

end # module ModalSolver
