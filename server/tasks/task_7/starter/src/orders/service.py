from orders.models import LineItem, Order
from orders.serializers import serialize_order_summary


def build_order(order_id: str, customer_tier: str, raw_items: list[dict]) -> Order:
    items = [
        LineItem(
            sku=item["sku"],
            quantity=item["quantity"],
            unit_price_cents=item["unit_price_cents"],
        )
        for item in raw_items
    ]
    return Order(order_id=order_id, customer_tier=customer_tier, items=items)


def checkout(order_id: str, customer_tier: str, raw_items: list[dict]) -> dict:
    order = build_order(order_id, customer_tier, raw_items)
    return serialize_order_summary(order)
