from orders.models import Order
from orders.pricing import calculate_total_cents


def serialize_order_summary(order: Order) -> dict:
    return {
        "order_id": order.order_id,
        "customer_tier": order.customer_tier,
        "line_count": len(order.items),
        "subtotal_cents": order.subtotal_cents(),
        "total_cents": calculate_total_cents(order),
    }
