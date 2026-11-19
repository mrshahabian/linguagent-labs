"""Tests for 64 — Order processing with business rules."""

import importlib
import os
import unittest

m = importlib.import_module(os.environ.get("PY_TARGET", "exercises"))

class TestOrderModels(unittest.TestCase):
    def test_item_creation(self):
        item = m.Item(id="item_1", price_cents=1000)
        self.assertEqual(item.id, "item_1")
        self.assertEqual(item.price_cents, 1000)
        
    def test_item_immutability(self):
        item = m.Item(id="item_1", price_cents=1000)
        with self.assertRaises(AttributeError):
            item.price_cents = 1200  # type: ignore
            
    def test_order_item_creation(self):
        order_item = m.OrderItem(item_id="item_1", quantity=5)
        self.assertEqual(order_item.item_id, "item_1")
        self.assertEqual(order_item.quantity, 5)

    def test_order_creation(self):
        order_item = m.OrderItem(item_id="item_1", quantity=5)
        order = m.Order(id="ord_123", items=[order_item], coupon_code="SAVE10")
        self.assertEqual(order.id, "ord_123")
        self.assertEqual(order.items, [order_item])
        self.assertEqual(order.coupon_code, "SAVE10")


class TestPricing(unittest.TestCase):
    def setUp(self):
        # We must dynamically instantiate the models from the target module
        self.catalog = {
            "apple": m.Item("apple", 100),   
            "banana": m.Item("banana", 150), 
            "laptop": m.Item("laptop", 100000) 
        }

    def test_base_price_no_rules(self):
        order = m.Order(
            id="ord_1",
            items=[
                m.OrderItem("apple", 2),
                m.OrderItem("banana", 1)
            ]
        )
        total = m.price_order(order, self.catalog, [])
        self.assertEqual(total, 350)  

    def test_bulk_discount_applies(self):
        order = m.Order(
            id="ord_2",
            items=[m.OrderItem("apple", 3)]
        )
        total = m.price_order(order, self.catalog, [m.bulk_item_discount])
        self.assertEqual(total, 270)

    def test_bulk_discount_does_not_apply_under_threshold(self):
        order = m.Order(
            id="ord_3",
            items=[m.OrderItem("apple", 2)]
        )
        total = m.price_order(order, self.catalog, [m.bulk_item_discount])
        self.assertEqual(total, 200)

    def test_coupon_save10(self):
        order = m.Order(
            id="ord_4",
            items=[m.OrderItem("banana", 2)],
            coupon_code="SAVE10"
        )
        total = m.price_order(order, self.catalog, [m.coupon_discount])
        self.assertEqual(total, 270)

    def test_coupon_flat5_under_limit(self):
        order = m.Order(
            id="ord_5",
            items=[m.OrderItem("laptop", 1)],
            coupon_code="FLAT5"
        )
        total = m.price_order(order, self.catalog, [m.coupon_discount])
        self.assertEqual(total, 99500)

    def test_coupon_flat5_exceeds_total(self):
        order = m.Order(
            id="ord_6",
            items=[m.OrderItem("apple", 2)], 
            coupon_code="FLAT5"
        )
        total = m.price_order(order, self.catalog, [m.coupon_discount])
        self.assertEqual(total, 0)

    def test_chained_rules_order_matters(self):
        order = m.Order(
            id="ord_7",
            items=[m.OrderItem("banana", 4)],
            coupon_code="SAVE10"
        )
        total = m.price_order(order, self.catalog, [m.bulk_item_discount, m.coupon_discount])
        self.assertEqual(total, 486)


class TestInventory(unittest.TestCase):
    def test_successful_reservation(self):
        inv = m.Inventory({"item_a": 10, "item_b": 5})
        order = m.Order(
            id="ord_1",
            items=[m.OrderItem("item_a", 3), m.OrderItem("item_b", 2)]
        )
        self.assertTrue(inv.reserve(order))
        self.assertEqual(inv.get_stock("item_a"), 7)
        self.assertEqual(inv.get_stock("item_b"), 3)

    def test_atomic_reservation_failure_leaves_stock_untouched(self):
        inv = m.Inventory({"item_a": 10, "item_b": 1})
        order = m.Order(
            id="ord_2",
            items=[m.OrderItem("item_a", 5), m.OrderItem("item_b", 2)] 
        )
        self.assertFalse(inv.reserve(order))
        self.assertEqual(inv.get_stock("item_a"), 10)
        self.assertEqual(inv.get_stock("item_b"), 1)

    def test_reservation_aggregates_duplicate_items(self):
        inv = m.Inventory({"item_a": 5})
        order = m.Order(
            id="ord_3",
            items=[m.OrderItem("item_a", 3), m.OrderItem("item_a", 3)] 
        )
        self.assertFalse(inv.reserve(order))
        self.assertEqual(inv.get_stock("item_a"), 5)

    def test_release_stock(self):
        inv = m.Inventory({"item_a": 5})
        order = m.Order(
            id="ord_4",
            items=[m.OrderItem("item_a", 2), m.OrderItem("item_a", 1)]
        )
        inv.release(order)
        self.assertEqual(inv.get_stock("item_a"), 8)


class TestOrderProcessing(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "item_a": m.Item("item_a", 1000),
            "item_b": m.Item("item_b", 2000)
        }
        self.inventory = m.Inventory({"item_a": 5, "item_b": 2})
        self.rules = [m.bulk_item_discount, m.coupon_discount]

    def test_process_order_success(self):
        order = m.Order(
            id="ord_success",
            items=[m.OrderItem("item_a", 2)],
            coupon_code="SAVE10"
        )
        result = m.process_order(order, self.catalog, self.inventory, self.rules)
        self.assertIsInstance(result, m.Receipt)
        self.assertEqual(result.order_id, "ord_success")
        self.assertEqual(result.total_cents, 1800)  
        self.assertEqual(self.inventory.get_stock("item_a"), 3)

    def test_process_order_missing_catalog_item(self):
        order = m.Order(
            id="ord_fail_catalog",
            items=[m.OrderItem("item_c", 1)]
        )
        result = m.process_order(order, self.catalog, self.inventory, self.rules)
        self.assertIsInstance(result, m.Rejection)
        self.assertEqual(result.reason, "Item item_c not found in catalog")
        self.assertEqual(self.inventory.get_stock("item_a"), 5)

    def test_process_order_invalid_quantity(self):
        order = m.Order(
            id="ord_fail_qty",
            items=[m.OrderItem("item_a", 0)]
        )
        result = m.process_order(order, self.catalog, self.inventory, self.rules)
        self.assertIsInstance(result, m.Rejection)
        self.assertEqual(result.reason, "Invalid quantity for item item_a")

    def test_process_order_insufficient_stock(self):
        order = m.Order(
            id="ord_fail_stock",
            items=[m.OrderItem("item_b", 3)]
        )
        result = m.process_order(order, self.catalog, self.inventory, self.rules)
        self.assertIsInstance(result, m.Rejection)
        self.assertEqual(result.reason, "Insufficient stock")
        self.assertEqual(self.inventory.get_stock("item_b"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
