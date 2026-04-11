from dataclasses import dataclass, field


@dataclass
class LineItem:
    sku: str
    quantity: int
    unit_price_cents: int


@dataclass
class Order:
    order_id: str
    customer_tier: str
    items: list[LineItem] = field(default_factory=list)

    def subtotal_cents(self) -> int:
        return sum(item.quantity * item.unit_price_cents for item in self.items)
