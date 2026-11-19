"""64 — Order processing with business rules

Reference implementation for the order processing system.
"""

from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, Union

@dataclass(frozen=True)
class Item:
    id: str
    price_cents: int

@dataclass(frozen=True)
class OrderItem:
    item_id: str
    quantity: int

@dataclass(frozen=True)
class Order:
    id: str
    items: List[OrderItem]
    coupon_code: Optional[str] = None


def bulk_item_discount(order: Order, catalog: Dict[str, Item], current_total: int) -> int:
    discount = 0
    for item in order.items:
        if item.quantity >= 3:
            if item.item_id in catalog:
                item_price = catalog[item.item_id].price_cents
                item_total = item_price * item.quantity
                discount += item_total // 10
    return discount

def coupon_discount(order: Order, catalog: Dict[str, Item], current_total: int) -> int:
    if not order.coupon_code:
        return 0
    if order.coupon_code == "SAVE10":
        return current_total // 10
    if order.coupon_code == "FLAT5":
        return min(500, current_total)
    return 0

def price_order(order: Order, catalog: Dict[str, Item], rules: List[Callable[[Order, Dict[str, Item], int], int]]) -> int:
    base_total = 0
    for item in order.items:
        if item.item_id in catalog:
            base_total += catalog[item.item_id].price_cents * item.quantity
            
    running_total = base_total
    for rule in rules:
        discount = rule(order, catalog, running_total)
        running_total = max(0, running_total - discount)
        
    return running_total


class Inventory:
    def __init__(self, stock: Dict[str, int]):
        self._stock = dict(stock)
        
    def reserve(self, order: Order) -> bool:
        requirements: Dict[str, int] = {}
        for item in order.items:
            requirements[item.item_id] = requirements.get(item.item_id, 0) + item.quantity
            
        for item_id, qty in requirements.items():
            if self._stock.get(item_id, 0) < qty:
                return False
                
        for item_id, qty in requirements.items():
            self._stock[item_id] -= qty
        return True
        
    def release(self, order: Order) -> None:
        requirements: Dict[str, int] = {}
        for item in order.items:
            requirements[item.item_id] = requirements.get(item.item_id, 0) + item.quantity
            
        for item_id, qty in requirements.items():
            self._stock[item_id] = self._stock.get(item_id, 0) + qty
            
    def get_stock(self, item_id: str) -> int:
        return self._stock.get(item_id, 0)


@dataclass(frozen=True)
class Receipt:
    order_id: str
    total_cents: int

@dataclass(frozen=True)
class Rejection:
    order_id: str
    reason: str

def process_order(
    order: Order,
    catalog: Dict[str, Item],
    inventory: Inventory,
    rules: List[Callable[[Order, Dict[str, Item], int], int]]
) -> Union[Receipt, Rejection]:
    for item in order.items:
        if item.item_id not in catalog:
            return Rejection(order.id, f"Item {item.item_id} not found in catalog")
        if item.quantity <= 0:
            return Rejection(order.id, f"Invalid quantity for item {item.item_id}")
            
    if not inventory.reserve(order):
        return Rejection(order.id, "Insufficient stock")
        
    total = price_order(order, catalog, rules)
    return Receipt(order.id, total)
