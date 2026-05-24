CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id SERIAL PRIMARY KEY,
    run_date DATE UNIQUE NOT NULL,
    rmse_solar_mw FLOAT,
    rmse_wind_mw FLOAT,
    rmse FLOAT,
    n_uur INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    tijd TIMESTAMPTZ NOT NULL,
    run_date DATE NOT NULL,
    solar_mw_pred FLOAT,
    wind_mw_pred FLOAT,
    solar_mw_actual FLOAT,
    wind_mw_actual FLOAT,
    UNIQUE (tijd, run_date)
);
