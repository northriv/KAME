/*
 * xnode_typename_test.cpp
 *
 * Regression test for the template-alias getTypename() bug.
 *
 * Problem: XGraphMathToolX<F> is a template, and its type aliases
 * (e.g. XGraph1DMathToolSum = XGraph1DMathToolX<FuncGraph1DMathToolSum>)
 * lose their human-readable name at runtime.  typeid(*this).name() returns
 * the mangled C++ name of the *template instantiation* — it is completely
 * unaware of the typedef/using alias.  When the ruby serializer calls
 * getTypename() to write a .kam file, it records the mangled name instead of
 * the registered key ("Graph1DMathToolSum"), and on load createByTypename()
 * cannot find a matching creator, so the tool is silently dropped.
 *
 * Fix: createByTypename() calls setStoredTypename(key) immediately after
 * construction, and getTypename() returns that stored string when non-empty.
 *
 * This test is deliberately self-contained (no Qt, no XNode) so it can run
 * as part of the lightweight tests/ suite.
 */

#include <cassert>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>

// ── Minimal node with the typename pattern ───────────────────────────────────

class SimpleNode {
public:
    virtual ~SimpleNode() {}

    //! Default: returns typeid name — mangled for template instantiations.
    virtual std::string getTypename() const {
        return typeid(*this).name();
    }

    //! Returns the stored override when set, raw typeid otherwise.
    //! This mirrors XGraphMathTool::getTypename().
    std::string getEffectiveTypename() const {
        return m_storedTypename.empty() ? getTypename() : m_storedTypename;
    }
    void setStoredTypename(const std::string &t) { m_storedTypename = t; }

private:
    std::string m_storedTypename;
};

// ── Template functor types mirroring XGraph1DMathToolX<F> ───────────────────

struct FuncSum {
    double operator()(double a, double b) { return a + b; }
};
struct FuncAvg {
    double operator()(double a, double b) { return (a + b) / 2.0; }
};

template <class F>
class ToolX : public SimpleNode {
public:
    F functor;
};

using ToolSum = ToolX<FuncSum>;
using ToolAvg = ToolX<FuncAvg>;

// ── Minimal type-holder / createByTypename registry ──────────────────────────

struct TypeRegistry {
    using CreatorFn = std::function<std::shared_ptr<SimpleNode>()>;

    template <class T>
    void registerType(const std::string &key) {
        m_creators[key] = []() -> std::shared_ptr<SimpleNode> {
            return std::make_shared<T>();
        };
    }

    //! Mirrors XGraph1DMathToolList::createByTypename():
    //!   creates the node and immediately stores the lookup key.
    std::shared_ptr<SimpleNode> createByTypename(const std::string &key,
                                                  const std::string &/*name*/) {
        auto it = m_creators.find(key);
        if (it == m_creators.end())
            return {};
        auto node = it->second();
        if (node)
            node->setStoredTypename(key);
        return node;
    }

private:
    std::map<std::string, CreatorFn> m_creators;
};

// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_typeid_is_not_alias_name() {
    // typeid of a template instantiation is implementation-defined but is
    // guaranteed NOT to be the plain typedef name.  Confirm this is the root
    // cause we need to defend against.
    ToolSum node;
    std::string raw = node.getTypename();
    printf("typeid raw name for ToolX<FuncSum>: %s\n", raw.c_str());
    assert(raw != "ToolSum" &&
           "Unexpected: typeid returned the typedef alias name — this is a "
           "compiler oddity; the test assumption no longer holds.");
    printf("PASS: typeid is NOT the typedef alias name\n");
}

static void test_stored_typename_overrides_typeid() {
    ToolSum node;
    // Before storing: raw typeid name used.
    assert(node.getEffectiveTypename() == node.getTypename());
    // After storing: stored name returned.
    node.setStoredTypename("ToolSum");
    assert(node.getEffectiveTypename() == "ToolSum");
    assert(node.getTypename() != "ToolSum"); // raw typeid still mangled
    printf("PASS: setStoredTypename overrides getEffectiveTypename\n");
}

static void test_createByTypename_unknown_returns_null() {
    TypeRegistry reg;
    reg.registerType<ToolSum>("ToolSum");
    auto node = reg.createByTypename("NoSuchTool", "n");
    assert(!node);
    printf("PASS: createByTypename with unknown key returns null\n");
}

static void test_createByTypename_sets_stored_typename() {
    TypeRegistry reg;
    reg.registerType<ToolSum>("ToolSum");
    reg.registerType<ToolAvg>("ToolAvg");

    auto sum = reg.createByTypename("ToolSum", "s");
    assert(sum);
    // Key was stored — serializer will write correct name.
    assert(sum->getEffectiveTypename() == "ToolSum");
    // Raw typeid is still the mangled template name.
    assert(sum->getTypename() != "ToolSum");

    auto avg = reg.createByTypename("ToolAvg", "a");
    assert(avg);
    assert(avg->getEffectiveTypename() == "ToolAvg");

    // Both are ToolX<> template instances; they differ only by stored name.
    assert(sum->getEffectiveTypename() != avg->getEffectiveTypename());

    printf("PASS: createByTypename sets stored typename correctly\n");
}

static void test_serialization_roundtrip_via_stored_typename() {
    // Simulate save→load cycle:
    //   save:  record getEffectiveTypename() → "ToolSum"
    //   load:  pass that string back to createByTypename()
    TypeRegistry reg;
    reg.registerType<ToolSum>("ToolSum");

    auto original = reg.createByTypename("ToolSum", "tool1");
    assert(original);

    // Save step: serializer captures the typename.
    std::string saved_typename = original->getEffectiveTypename();

    // Load step: createByTypename is called with the saved string.
    auto loaded = reg.createByTypename(saved_typename, "tool1");
    assert(loaded && "Load failed — saved typename not recognized by registry");
    assert(loaded->getEffectiveTypename() == saved_typename);

    printf("PASS: serialization roundtrip via stored typename\n");
}

static void test_unregistered_typename_fails_roundtrip() {
    // If we accidentally save the raw typeid name (bug state), loading fails.
    TypeRegistry reg;
    reg.registerType<ToolSum>("ToolSum");

    ToolSum node_without_fix;
    // Bug state: getEffectiveTypename() falls back to mangled typeid.
    std::string buggy_saved = node_without_fix.getTypename(); // mangled

    auto loaded = reg.createByTypename(buggy_saved, "tool1");
    assert(!loaded && "Bug regression: mangled name should NOT match any registered type");

    printf("PASS: raw typeid name does not match registered key (confirms bug scenario)\n");
}

int main() {
    test_typeid_is_not_alias_name();
    test_stored_typename_overrides_typeid();
    test_createByTypename_unknown_returns_null();
    test_createByTypename_sets_stored_typename();
    test_serialization_roundtrip_via_stored_typename();
    test_unregistered_typename_fails_roundtrip();
    printf("All xnode_typename tests passed.\n");
    return 0;
}
