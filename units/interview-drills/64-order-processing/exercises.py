"""64 — Order processing with business rules

Implement the classes and functions below so that the tests in test_exercises.py pass.
Read README.md first for detailed instructions.
"""

from typing import Callable, List, Dict, Optional, Union
from dataclasses import dataclass

# --- Step 1: Domain Models ---
# Note: The learner must implement these as frozen dataclasses.

class Item:
    """An immutable product in the catalog.
    
    Should be implemented as a frozen dataclass.
    
    Attributes:
        id: Unique identifier for the item.
        price_cents: Price of a single unit in integer cents.
    """
    def __init__(self, id: str, price_cents: int):
        raise NotImplementedError("your code here")

class OrderItem:
    """An item and quantity requested in an order.
    
    Should be implemented as a frozen dataclass.
    
    Attributes:
        item_id: The ID of the item being ordered.
        quantity: How many units are requested.
    """
    def __init__(self, item_id: str, quantity: int):
        raise NotImplementedError("your code here")

class Order:
    """A customer order containing one or more items and an optional coupon code.
    
    Should be implemented as a frozen dataclass.
    
    Attributes:
        id: Unique identifier for the order.
        items: List of OrderItem instances.
        coupon_code: Optional string code representing a discount coupon.
    """
    def __init__(self, id: str, items: List[OrderItem], coupon_code: Optional[str] = None):
        raise NotImplementedError("your code here")


# --- Step 2: Pricing Rules ---

PricingRule = Callable[[Order, Dict[str, Item], int], int]

def bulk_item_discount(order: Order, catalog: Dict[str, Item], current_total: int) -> int:
    """Calculate bulk discount.
    
    Rule: For any OrderItem with a quantity of 3 or more, apply a 10% discount
    to that item's portion of the price. Round the discount down to the nearest integer cent.
    """
    raise NotImplementedError("your code here")

def coupon_discount(order: Order, catalog: Dict[str, Item], current_total: int) -> int:
    """Calculate coupon discount based on the order's coupon code.
    
    Rules:
    - If coupon_code is "SAVE10", apply a 10% discount to the current running total
      (rounded down to the nearest integer cent).
    - If coupon_code is "FLAT5", apply a flat 500 cents ($5.00) discount.
      The discount cannot exceed the current running total (total cannot go below 0).
    - Any other coupon code yields 0 discount.
    """
    raise NotImplementedError("your code here")

def price_order(order: Order, catalog: Dict[str, Item], rules: List[PricingRule]) -> int:
    """Calculate the final price of an order in cents.
    
    First, calculate the base total (sum of item price * quantity for all items).
    Then, apply each pricing rule in the order they are provided in the `rules` list.
    Each rule calculates its discount based on the *current running total* after previous
    rules have been applied. Subtract each discount from the running total.
    
    The final total cannot be negative (if discounts exceed the price, return 0).
    """
    raise NotImplementedError("your code here")


# --- Step 3: Atomic Inventory Management ---

class Inventory:
    """Manages stock levels of items in memory with atomic operations."""
    
    def __init__(self, stock: Dict[str, int]):
        """Initialize inventory with a dictionary mapping item_id -> quantity available.
        
        Make a copy of the stock dictionary to prevent external mutation.
        """
        raise NotImplementedError("your code here")
        
    def reserve(self, order: Order) -> bool:
        """Attempt to reserve stock for an entire order atomically.
        
        If the inventory has sufficient stock for ALL items in the order, deduct
        the quantities from the stock and return True.
        
        If any item in the order is short of stock, make NO changes to the inventory
        at all (all-or-nothing rollback) and return False.
        
        Note: An order may contain multiple OrderItems with the same item_id.
        You must aggregate the requested quantities before checking stock.
        """
        raise NotImplementedError("your code here")
        
    def release(self, order: Order) -> None:
        """Release reserved stock back to the inventory.
        
        Add the quantities of all items in the order back to the stock levels.
        Aggregate quantities if the order contains duplicate item IDs.
        """
        raise NotImplementedError("your code here")
        
    def get_stock(self, item_id: str) -> int:
        """Return the current stock level for an item. Return 0 if item is not tracked."""
        raise NotImplementedError("your code here")


# --- Step 4: Orchestration & Results ---

class Receipt:
    """Returned on successful order processing.
    
    Should be implemented as a frozen dataclass.
    
    Attributes:
        order_id: Unique identifier for the order.
        total_cents: The final price paid in cents.
    """
    def __init__(self, order_id: str, total_cents: int):
        raise NotImplementedError("your code here")

class Rejection:
    """Returned when an order cannot be processed.
    
    Should be implemented as a frozen dataclass.
    
    Attributes:
        order_id: Unique identifier for the order.
        reason: Human-readable reason for rejection.
    """
    def __init__(self, order_id: str, reason: str):
        raise NotImplementedError("your code here")

def process_order(
    order: Order,
    catalog: Dict[str, Item],
    inventory: Inventory,
    rules: List[PricingRule]
) -> Union[Receipt, Rejection]:
    """Orchestrate the order processing pipeline.
    
    Steps:
    1. Validation:
       - Verify that every item in the order exists in the catalog.
         If not, return a Rejection with reason: "Item {item_id} not found in catalog"
         (use the first invalid item ID found in the order's item list order).
       - Verify that every item in the order has a quantity greater than 0.
         If not, return a Rejection with reason: "Invalid quantity for item {item_id}"
         (use the first item ID with invalid quantity found).
    2. Inventory Reservation:
       - Attempt to reserve stock atomically using the inventory.
         If reservation fails, return a Rejection with reason: "Insufficient stock".
    3. Pricing:
       - Calculate the final price using `price_order` and the provided rules.
    4. Success:
       - Return a Receipt with the order ID and the final total in cents.
    """
    raise NotImplementedError("your code here")
