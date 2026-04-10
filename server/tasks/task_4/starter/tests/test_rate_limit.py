from src.rate_limit import RateLimiter


def test_allows_requests_up_to_limit():
    limiter = RateLimiter(limit=2, window_seconds=10)
    assert limiter.allow(0) is True
    assert limiter.allow(1) is True
    assert limiter.allow(2) is False


def test_old_requests_expire_out_of_window():
    limiter = RateLimiter(limit=2, window_seconds=10)
    assert limiter.allow(0) is True
    assert limiter.allow(5) is True
    assert limiter.allow(11) is True


def test_exact_boundary_is_treated_as_expired():
    limiter = RateLimiter(limit=1, window_seconds=10)
    assert limiter.allow(3) is True
    assert limiter.allow(13) is True
