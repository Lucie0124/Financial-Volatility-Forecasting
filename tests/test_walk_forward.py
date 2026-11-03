import pandas as pd
from market_risk.modelling.dataset import create_walk_forward_fold
from market_risk.features.build_target import build_target

# pytest test_walk_forward.py -v 


# Test that train data always comes before test data
def test_train_dates_are_before_test_dates():
    dates = pd.bdate_range(start="2021-01-01", end="2022-12-31")

    df = pd.DataFrame({
        "date": dates,
        "dummy_feature": range(len(dates)),
    })

    train, test = create_walk_forward_fold(
        df=df,
        test_year=2022,
    )

    assert train["date"].max() < test["date"].min()

# Test that the test set contains only the requested year
def test_test_set_contains_only_requested_year():
    dates = pd.bdate_range(
        start="2020-01-01",
        end="2023-12-31",
    )

    df = pd.DataFrame({
        "date": dates,
        "dummy_feature": range(len(dates)),
    })

    _, test = create_walk_forward_fold(
        df=df,
        test_year=2022,
    )

    assert (test["date"].dt.year == 2022).all()
    
# Test the five-day purge
def test_five_day_purge():
    dates = pd.bdate_range(
        start="2021-01-01",
        end="2022-12-31",
    )

    df = pd.DataFrame({
        "date": dates,
        "dummy_feature": range(len(dates)),
    })

    train, test = create_walk_forward_fold(
        df=df,
        test_year=2022,
    )

    first_test_date = test["date"].min()

    pre_test_data = df[
        df["date"] < first_test_date
    ].sort_values("date")

    expected_last_train_date = pre_test_data.iloc[-6]["date"]

    assert train["date"].max() == expected_last_train_date
    
# Test that exactly five observations are removed
def test_exactly_five_rows_are_purged():
    dates = pd.bdate_range(
        start="2021-01-01",
        end="2022-12-31",
    )

    df = pd.DataFrame({
        "date": dates,
        "dummy_feature": range(len(dates)),
    })

    train, test = create_walk_forward_fold(
        df=df,
        test_year=2022,
    )

    first_test_date = test["date"].min()

    all_possible_train_rows = df[
        df["date"] < first_test_date
    ]

    assert len(all_possible_train_rows) - len(train) == 5

# Test expanding-window behaviour
def test_training_window_expands():
    dates = pd.bdate_range(
        start="2020-01-01",
        end="2023-12-31",
    )

    df = pd.DataFrame({
        "date": dates,
        "dummy_feature": range(len(dates)),
    })

    train_2022, _ = create_walk_forward_fold(
        df=df,
        test_year=2022,
    )

    train_2023, _ = create_walk_forward_fold(
        df=df,
        test_year=2023,
    )

    assert len(train_2023) > len(train_2022)

    assert train_2023["date"].min() == train_2022["date"].min()


def test_purged_training_targets_do_not_use_test_period_returns():
    dates = pd.bdate_range("2021-10-01", "2022-02-01")
    df = build_target(pd.DataFrame({"date": dates, "return_1d": 0.01}))
    train, test = create_walk_forward_fold(df, test_year=2022)
    changed = df.copy()
    changed.loc[changed["date"] >= test["date"].min(), "return_1d"] = 0.90

    changed_targets = build_target(changed)

    pd.testing.assert_series_equal(
        train["target_volatility_5d"],
        changed_targets.loc[train.index, "target_volatility_5d"],
    )


def test_cleaning_sorts_before_forward_filling(tmp_path, monkeypatch):
    from market_risk.data.prepare import clean_data, INDICATOR_COLUMNS
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-05", "2022-01-03", "2022-01-04"]),
        "sp500": [102.0, 100.0, 101.0],
    })
    for column in INDICATOR_COLUMNS:
        frame[column] = [99.0, 10.0, float("nan")]
    path = tmp_path / "market.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr("market_risk.data.prepare.INPUT_PATH", path)
    cleaned = clean_data()
    assert cleaned["date"].is_monotonic_increasing
    assert cleaned.loc[1, "vix"] == 10.0
