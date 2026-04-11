from orders.models import Order


DISCOUNT_BY_TIER = {
    "standard": 0,
    "gold": 10,
    "platinum": 15,
}


def shipping_cents(order: Order) -> int:
    subtotal = order.subtotal_cents()
    if subtotal >= 5000:
        return 0
    return 799


def calculate_total_cents(order: Order) -> int:
    subtotal = order.subtotal_cents()
    discount_percent = DISCOUNT_BY_TIER.get(order.customer_tier, 0)
    discount_cents = subtotal * discount_percent
    return subtotal - discount_cents + shipping_cents(order)
