# KAME Python binding tests
# Open this file in KAME's tabbed terminal to run interactively.

import kame
from kame import *

def main():
    # ── helpers ───────────────────────────────────────────────────────────────

    def _assert_eq(a, b, msg=""):
        if a != b:
            raise AssertionError(f"{msg}: expected {b!r}, got {a!r}")

    def _assert(cond, msg="assertion failed"):
        if not cond:
            raise AssertionError(msg)

    def _pass(name):
        print(f"  PASS: {name}")

    def _skip(name, reason=""):
        print(f"  SKIP: {name}" + (f" — {reason}" if reason else ""))

    # ── individual tests ──────────────────────────────────────────────────────

    def test_xnode_get_typename():
        """XNode.getTypename() strips the leading 'X' prefix."""
        node = XStringNode("t", False)
        _assert_eq(node.getTypename(), "StringNode", "XStringNode.getTypename()")
        _pass("XNode.getTypename()")

    def test_xnode_get_label():
        """Node label matches the name passed to the constructor."""
        node = XStringNode("my_label", False)
        _assert_eq(node.getLabel(), "my_label", "getLabel()")
        _pass("XNode.getLabel()")

    def test_node_set_and_read():
        """node.set(v) + float(node)/int(node) round-trip — the standard pattern from pydrivers.py."""
        # XDoubleNode
        dnode = XDoubleNode("dbl", False)
        dnode.set(3.14)
        _assert(abs(float(dnode) - 3.14) < 1e-12, f"XDoubleNode: expected 3.14, got {float(dnode)}")
        # XIntNode
        inode = XIntNode("int", False)
        inode.set(42)
        _assert_eq(int(inode), 42, "XIntNode")
        # XBoolNode
        bnode = XBoolNode("boo", False)
        bnode.set(True)
        _assert(bool(bnode), "XBoolNode set True")
        _pass("node.set + float/int/bool read")

    def test_node_child_setitem():
        """parent['child'] = val sets child value via internal trans() — mirrors pydrivers.py."""
        parent = XDoubleNode("parent", False)
        child  = XDoubleNode("child",  False)
        parent.insert(child)
        parent["child"] = 2.71
        _assert(abs(float(child) - 2.71) < 1e-12,
                f"child after parent['child']=2.71: got {float(child)}")
        _pass("node['child'] = val")

    def test_xstringlist_create_by_typename():
        """XStringList.createByTypename always produces an XStringNode."""
        try:
            lst = kame.XStringList("sl", False)
        except AttributeError:
            _skip("XStringList.createByTypename", "XStringList not bound in this build")
            return
        node = lst.createByTypename("ignored", "item")
        _assert(node is not None, "createByTypename returned None")
        _pass("XStringList.createByTypename")

    def test_math_tool_stored_typename(driver_name):
        """
        After createByTypename("Graph1DMathToolSum", ...),
        tool.getTypename() must return "Graph1DMathToolSum" — not a mangled C++ name.
        This is the core regression for .kam save/load.
        """
        try:
            driver = Root()["Drivers"][driver_name]
        except Exception:
            driver = None
        if not driver:
            _skip("math tool stored typename", f"driver '{driver_name}' not found")
            return
        toollist = None
        for candidate in ["Wave-Math", "Spectrum-Math", "Plot1-Math"]:
            try:
                toollist = driver[candidate]
            except Exception:
                pass
            if toollist:
                break
        if not toollist:
            _skip("math tool stored typename", f"no *-Math child under '{driver_name}'")
            return
        tool = toollist.createByTypename("Graph1DMathToolSum", "sum_test")
        _assert(tool is not None, "createByTypename('Graph1DMathToolSum') failed")
        _assert_eq(tool.getTypename(), "Graph1DMathToolSum",
                   "typename must be the registered key, not a mangled C++ name")
        _pass("math tool stored typename roundtrip")

    # ── menu ──────────────────────────────────────────────────────────────────

    unit_tests = [
        ("XNode.getTypename()",          test_xnode_get_typename),
        ("XNode.getLabel()",             test_xnode_get_label),
        ("node.set + float/int/bool",    test_node_set_and_read),
        ("node['child'] = val",          test_node_child_setitem),
        ("XStringList.createByTypename", test_xstringlist_create_by_typename),
    ]

    def run_unit_tests():
        for name, fn in unit_tests:
            try:
                fn()
            except Exception as e:
                print(f"  FAIL: {name} — {e}")

    def run_integration(driver_name):
        try:
            test_math_tool_stored_typename(driver_name)
        except Exception as e:
            print(f"  FAIL: math tool stored typename — {e}")

    while True:
        print()
        print("── KAME Python Tests ─────────────────────────────")
        for i, (name, _) in enumerate(unit_tests, 1):
            print(f"  {i}. {name}")
        n = len(unit_tests)
        print(f"  {n+1}. [Integration] Math tool stored typename")
        print(f"  a. Run all unit tests")
        print(f"  A. Run all (unit + integration)")
        print(f"  q. Quit")
        print("──────────────────────────────────────────────────")
        choice = input("Choice: ").strip()

        if choice == "q":
            break
        elif choice == "a":
            run_unit_tests()
        elif choice == "A":
            run_unit_tests()
            driver_name = input("  Driver name for integration tests: ").strip()
            run_integration(driver_name)
        elif choice == str(n + 1):
            driver_name = input("  Driver name: ").strip()
            run_integration(driver_name)
        elif choice.isdigit() and 1 <= int(choice) <= n:
            _, fn = unit_tests[int(choice) - 1]
            try:
                fn()
            except Exception as e:
                print(f"  FAIL: {e}")
        else:
            print("  Unknown choice.")

main()
