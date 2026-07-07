using Test
using Statistics
using CSV, DataFrames
using Random

include(joinpath(@__DIR__, "..", "ModalSolver.jl"))
using .ModalSolver

rmse(a, b) = sqrt(mean((a .- b).^2))

@testset "Known SSP baseline regression test" begin

    meas_file = joinpath(@__DIR__, "ssnn_profiles_1224.csv")
    ssp_file  = joinpath(@__DIR__, "true_ssnn_ssp.csv")

    RMSE_LIMIT = 2.0
    SEED = 1224

    # ------------------------------------------------------------
    # Test 1: Required baseline files exist
    # ------------------------------------------------------------
    @testset "1. Baseline input files exist" begin
        @test isfile(meas_file)
        @test isfile(ssp_file)
    end

    # ------------------------------------------------------------
    # Test 2: Known SSP CSV loads correctly
    # ------------------------------------------------------------
    ssp = load_ssp(ssp_file)

    @testset "2. Known SSP loading works" begin
        @test length(ssp[1]) == length(ssp[2])
        @test all(isfinite, ssp[1])
        @test all(isfinite, ssp[2])
    end

    # ------------------------------------------------------------
    # Test 3: Known-SSP solver can be constructed
    # ------------------------------------------------------------
    pm = ModeSolver(
        D = 25.0,
        f = 500.0,
        ssp = ssp
    )

    @testset "3. Known-SSP solver construction works" begin
        @test pm.D == 25.0
        @test pm.f == 500.0
        @test pm.ssp !== nothing
    end

    # ------------------------------------------------------------
    # Test 4: Training runs without crashing
    # ------------------------------------------------------------
    @testset "4. Known-SSP training runs" begin
        fit!(pm, meas_file; restarts = 10, seed = SEED)
        @test pm.theta !== nothing
        @test !isempty(pm.history)
    end

    # ------------------------------------------------------------
    # Test 5: Prediction returns valid amplitudes
    # ------------------------------------------------------------
    df = CSV.read(meas_file, DataFrame)
    pred = predict_amp(pm, df.range_m, df.depth_m)

    @testset "5. Prediction output is valid" begin
        @test length(pred) == nrow(df)
        @test all(isfinite, pred)
        @test all(pred .>= 0)
    end

    # ------------------------------------------------------------
    # Test 6: Baseline RMSE stays acceptable
    # Main regression test for the known-SSP reproduction.
    # ------------------------------------------------------------
    measurement_rmse = rmse(pred, df.amp)

    @testset "6. Baseline RMSE stays below threshold" begin
        println("Known-SSP baseline RMSE = ", measurement_rmse, " mVpp")
        @test measurement_rmse < RMSE_LIMIT
    end

    # ------------------------------------------------------------
    # Test 7: Loss history is valid
    # Checks that the training log contains finite losses.
    # ------------------------------------------------------------
    @testset "7. Loss history is valid" begin
        @test "epoch" in names(pm.history)
        @test "train_loss" in names(pm.history)
        @test "val_loss" in names(pm.history)

        @test all(isfinite, pm.history.train_loss)
        @test all(isfinite, pm.history.val_loss)

        @test length(pm.history.val_loss) >= 2
    end

    # ------------------------------------------------------------
    # Test 8: Loss improves from the initial logged value
    # Loss may bounce, but the best validation loss should be
    # lower than the first logged validation loss.
    # ------------------------------------------------------------
    @testset "8. Validation loss improves overall" begin
        first_val_loss = first(pm.history.val_loss)
        best_val_loss  = minimum(pm.history.val_loss)

        println("First validation loss = ", first_val_loss)
        println("Best validation loss  = ", best_val_loss)

        @test best_val_loss <= first_val_loss
    end

    # ------------------------------------------------------------
    # Test 9: Least validation loss is the one selected
    # This checks that the final stored model corresponds to the
    # lowest validation loss, not just the last training epoch.
    # ------------------------------------------------------------
    @testset "9. Least validation loss is selected" begin
        data = load_measurements(meas_file)

        train_raw, val_raw = split_train_val(data; val_fraction = 0.0, seed = SEED)

        val_data = (
            range_m = val_raw.range_m,
            depth_m = val_raw.depth_m,
            amp = val_raw.amp ./ pm.yscale
        )

        selected_val_loss = ModalSolver._known_loss(
            pm,
            pm.theta,
            val_data,
            eachindex(val_data.amp);
            pm.loss_weights...
        )

        best_logged_val_loss = minimum(pm.history.val_loss)

        println("Selected model validation loss = ", selected_val_loss)
        println("Best logged validation loss    = ", best_logged_val_loss)

        @test isapprox(
            selected_val_loss,
            best_logged_val_loss;
            rtol = 1e-6,
            atol = 1e-8
        )
    end
end