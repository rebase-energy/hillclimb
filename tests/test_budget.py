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


def test_clock_str_is_fixed_shape_for_the_log_gutter():
    from hillclimb.harness.budget import format_clock

    assert format_clock(582) == "9:42"
    assert format_clock(600) == "10:00"
    assert format_clock(3900) == "1:05:00"
    assert format_clock(0) == "0:00" and format_clock(-3) == "0:00"
    assert BudgetManager(total_s=600, stop_margin_s=0).clock_str().startswith(("10:00", "9:5"))


def test_a_short_budget_stops_once_spent():
    """A budget under 10 s has a stop margin of 0; once it is spent the
    search must stop, not idle at 0:00 forever (remaining() is clamped at 0,
    so a strict `<` never fired)."""
    budget = BudgetManager(total_s=5, stop_margin_s=300, spent_s=6)
    assert budget.stop_margin_s == 0 and budget.remaining() == 0
    assert budget.should_stop()
