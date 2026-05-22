"""Tests voor de train module."""

import pandas as pd

from scripts.train import FEATURES, TARGETS, TRAIN_RATIO, make_splits


def create_dataframe():
    # 10 rijen zodat de 80/20-split minstens 1 rij in elke set geeft
    idx = pd.date_range("2022-01-01 00:00:00", periods=10, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "solar_mw": [0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.3, 0.8, 1.2, 0.6],
            "wind_mw": [50.0, 52.0, 48.0, 55.0, 51.0, 49.0, 53.0, 47.0, 54.0, 50.0],
            "wind_speed_kmh": [
                12.0,
                13.0,
                11.0,
                14.0,
                12.5,
                10.5,
                13.5,
                11.5,
                14.5,
                12.0,
            ],
            "solar_radiation_wm2": [
                0.0,
                0.0,
                50.0,
                80.0,
                60.0,
                0.0,
                0.0,
                60.0,
                90.0,
                30.0,
            ],
            "hour": idx.hour,
            "month": idx.month,
            "dayofweek": idx.dayofweek,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
        },
        index=idx,
    )


def test_make_splits_array_vormen():
    df = create_dataframe()
    x_train, y_train, x_val, y_val = make_splits(df)
    # x heeft 6 features, y heeft 2 doelvariabelen (solar_mw en wind_mw)
    assert x_train.shape[1] == len(FEATURES)
    assert y_train.shape[1] == len(TARGETS)
    assert x_val.shape[1] == len(FEATURES)
    assert y_val.shape[1] == len(TARGETS)


def test_make_splits_grootte():
    df = create_dataframe()
    x_train, _, x_val, _ = make_splits(df)
    # 80% train, 20% val van 10 rijen = 8 + 2
    assert len(x_train) == 8
    assert len(x_val) == 2


def test_make_splits_geen_overlap():
    # Train en validatie mogen nooit dezelfde tijdstempels delen
    df = create_dataframe()
    split_idx = int(len(df) * TRAIN_RATIO)
    train_idx = set(df.iloc[:split_idx].index)
    val_idx = set(df.iloc[split_idx:].index)
    assert train_idx.isdisjoint(val_idx)


def test_make_splits_val_na_train():
    # Validatiedata moet chronologisch na de traindata vallen
    df = create_dataframe()
    split_idx = int(len(df) * TRAIN_RATIO)
    assert df.iloc[:split_idx].index.max() < df.iloc[split_idx:].index.min()
