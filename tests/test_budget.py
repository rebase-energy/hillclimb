from hillclimb.harness.budget import BudgetManager


def test_remaining_and_stop():
    budget = BudgetManager(total_s=100, stop_margin_s=10)
    assert 99 < budget.remaining() <= 100
    assert not budget.should_stop()


def test_spent_seed():
    budget = BudgetManager(total_s=100, stop_margin_s=10, spent_s=95)
    assert budget.remaining() <= 5
    assert budget.should_stop()


def test_remaining_str():
    assert BudgetManager(total_s=7200, stop_margin_s=0).remaining_str().startswith("2h")
    assert "minutes" in BudgetManager(total_s=600, stop_margin_s=0).remaining_str()
