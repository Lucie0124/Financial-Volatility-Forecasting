import pandas as pd
from market_risk.modelling.dataset import create_walk_forward_fold

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