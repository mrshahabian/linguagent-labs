# 64 — Order processing with business rules

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrshahabian/linguagent-labs/blob/main/units/interview-drills/64-order-processing/lab.ipynb) [![Open in LinguAgent](https://img.shields.io/badge/Open%20in-LinguAgent-1f6f5c)](https://linguagent.com/courses/interview-drills/order-processing?utm_source=github&utm_medium=labs)

*~45 minutes · 19 tests · Interview*

In software engineering interviews, you are rarely asked to build a static system. Instead, interviewers love to start with a simple problem—like calculating an order's total—and then layer on changing business requirements mid-interview. They might say: "Now add a 10% discount for bulk items," followed by "Now add a coupon code," and finally "Now make sure we don't sell items we don't have in stock."

If you hardcode these rules into a single giant function, your code quickly becomes a tangled mess of nested `if/else` statements. A change to one rule breaks another, and the code becomes impossible to unit test in isolation. This module teaches you how to design a flexible, extensible order processing pipeline using clean domain models, the Strategy pattern for business rules, and atomic state transitions.

## What you'll build

We will build a robust order processing system that handles catalog validation, dynamic pricing rules, and atomic inventory reservations using the following components:

| Function / class | You'll implement | Shows up in real code as |
|---|---|---|
| `Item`, `OrderItem`, `Order` | Immutable domain models using dataclasses | Core business entities representing products and customer orders |
| `price_order` | A pricing engine that chains extensible discount rules | E-commerce checkout pricing calculators with dynamic promotions |
| `Inventory` | An in-memory stock manager with atomic reservation | Database transaction boundaries or lock-free inventory allocation |
| `process_order` | An orchestrator returning a `Receipt` or `Rejection` | API controllers coordinating database, payment, and fulfillment steps |

---

## 1. Modeling with Dataclasses & Integer Cents

When representing money in code, the most common beginner mistake is using floating-point numbers (`float`). Because floats cannot precisely represent base-10 fractions, you quickly run into precision errors:

```python
>>> 0.1 + 0.2
0.30000000000000004
```

In a production system, this leads to financial discrepancies. The industry standard is to represent currency as **integer cents** (e.g., `$10.50` is represented as `1050`). 

Additionally, domain models should be immutable whenever possible. Using Python's `@dataclass(frozen=True)` ensures that once an `Item` or `Order` is created, it cannot be accidentally modified by a downstream function. This eliminates a massive class of state-mutation bugs.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Item:
    id: str
    price_cents: int
```

**In `exercises.py`:** Implement the `Item`, `OrderItem`, and `Order` dataclasses.

**Test it:** `python3 -m unittest test_exercises.TestOrderModels -v`

---

## 2. The Open-Closed Principle for Pricing Rules

If you write your pricing logic like this:

```python
def price_order(order, catalog):
    total = 0
    for item in order.items:
        total += catalog[item.item_id].price_cents * item.quantity
    if order.coupon_code == "SAVE10":
        total *= 0.9
    # What happens when we add bulk discounts? Or seasonal sales? 
    # This function will grow forever and violate the Single Responsibility Principle.
```

You violate the **Open-Closed Principle**: software entities should be open for extension, but closed for modification. 

Instead, we can define a pricing engine that takes a list of **rule functions**. Each rule function takes the order, the catalog, and the *current running total*, and returns the discount amount to subtract. This allows us to add new discount rules (like bulk discounts or coupon codes) without changing the core `price_order` function.

```python
def bulk_item_discount(order, catalog, current_total):
    # If quantity of any item is >= 3, get 10% off that item's portion of the price
    ...
```

**In `exercises.py`:** Implement `price_order`, `bulk_item_discount`, and `coupon_discount`.

**Test it:** `python3 -m unittest test_exercises.TestPricing -v`

---

## 3. Atomic State Changes (All-or-Nothing Inventory)

Imagine a customer orders 2 books and 1 laptop. We have 5 books in stock, but 0 laptops. If we deduct the 2 books from our inventory first, and then realize we don't have the laptop, we have to rollback the book deduction. If we forget to roll back, we leak inventory.

This is called an **atomic state transition**. When reserving stock for an order, we must verify that *all* items are available in their requested quantities *before* we deduct any stock. If even a single item is unavailable, we must make no changes to the inventory state at all.

```python
# Pitfall: Deducting on the fly
for item in order.items:
    if self.stock[item.item_id] >= item.quantity:
        self.stock[item.item_id] -= item.quantity # Dangerous if a later item fails!
```

Instead, perform a read-only check first, and only perform the write phase if all checks pass.

**In `exercises.py`:** Implement the `Inventory` class with atomic `reserve` and `release` methods.

**Test it:** `python3 -m unittest test_exercises.TestInventory -v`

---

## 4. Orchestration and the Result Type Pattern

In many languages, expected business failures (like "out of stock" or "invalid coupon") are handled by raising and catching exceptions. However, exceptions should be reserved for *exceptional* circumstances (like database connection failures). For expected business outcomes, returning a structured **Result** object is much cleaner and self-documenting.

We will define two result dataclasses:
- `Receipt`: Returned on success, containing the order ID and the final calculated total in cents.
- `Rejection`: Returned on failure, containing the order ID and a clear, human-readable reason string.

Our orchestrator `process_order` will tie everything together:
1. Validate that all items in the order exist in the catalog and have positive quantities (> 0).
2. Attempt to reserve inventory atomically.
3. Calculate the final price using our pricing rules.
4. Return a `Receipt` on success, or a `Rejection` on any failure.

**In `exercises.py`:** Implement `Receipt`, `Rejection`, and the `process_order` orchestrator.

**Test it:** `python3 -m unittest test_exercises.TestOrderProcessing -v`

---

## The exercises

| Class / Function | What it teaches |
|---|---|
| `Item`, `OrderItem`, `Order` | Modeling domain entities safely with frozen dataclasses and integer cents |
| `price_order` | Designing an extensible pricing pipeline using the Strategy pattern |
| `bulk_item_discount` | Implementing item-specific conditional pricing rules |
| `coupon_discount` | Implementing order-wide conditional discount rules |
| `Inventory` | Writing atomic state modification logic with rollback guarantees |
| `process_order` | Coordinating multiple domain services and returning explicit Result types |

## Check your understanding

1. **Why do we represent prices as integers (cents) instead of floats?**
   <details><summary>Answer</summary>Floating-point numbers use binary representation, which cannot precisely represent decimal fractions like 0.1 or 0.2. This leads to rounding errors over multiple operations. Integers represent exact values, eliminating rounding discrepancies.</details>

2. **What is the benefit of making domain models like `Item` and `Order` frozen?**
   <details><summary>Answer</summary>Frozen dataclasses are immutable. This prevents bugs where a pricing rule or inventory check accidentally modifies an item's price or an order's quantity during processing, ensuring data integrity throughout the pipeline.</details>

3. **How does the Strategy pattern in `price_order` make the system easier to extend?**
   <details><summary>Answer</summary>By accepting a list of rule callables, <code>price_order</code> doesn't need to know the details of any specific discount. If a new rule is introduced (e.g., "Buy One Get One Free"), we can write it as a standalone function and pass it in without modifying <code>price_order</code>'s code.</details>

4. **What does it mean for the `Inventory.reserve` method to be atomic?**
   <details><summary>Answer</summary>It means the reservation is "all-or-nothing". If the inventory has enough stock for every single item in the order, all items are deducted. If even one item is short, no stock is deducted for any item, leaving the system in a clean, consistent state.</details>

5. **Why return a `Rejection` object instead of raising an exception when stock is insufficient?**
   <details><summary>Answer</summary>Insufficient stock is a standard, expected business outcome, not an exceptional system failure. Returning a <code>Rejection</code> makes the control flow explicit, type-safe, and easier to follow without the overhead of try-except blocks.</details>

## Interview questions this module answers

- *How do you handle monetary calculations in software to prevent precision loss?*
- *How would you design a checkout system where discount rules change frequently?*
- *What is the Open-Closed Principle, and how have you applied it in your code?*
- *How do you ensure atomic state updates in memory without relying on a database transaction?*
- *What are the benefits of returning Result objects over throwing exceptions for business logic failures?*
