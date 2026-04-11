from orders.pricing import calculate_total_cents
from orders.service import build_order, checkout


def test_checkout_for_standard_customer_includes_shipping():
    summary = checkout(
        "ord-100",
        "standard",
        [{"sku": "A", "quantity": 1, "unit_price_cents": 1200}],
    )
    assert summary["subtotal_cents"] == 1200
    assert summary["total_cents"] == 1999


def test_gold_customer_receives_percentage_discount():
    order = build_order(
        "ord-101",
        "gold",
        [{"sku": "B", "quantity": 2, "unit_price_cents": 2000}],
    )
    assert calculate_total_cents(order) == 4399


def test_free_shipping_threshold_is_applied_after_discount_logic():
    summary = checkout(
        "ord-102",
        "platinum",
        [{"sku": "C", "quantity": 1, "unit_price_cents": 6000}],
    )
    assert summary["subtotal_cents"] == 6000
    assert summary["total_cents"] == 5100
