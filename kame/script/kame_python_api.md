# KAME Python API Quick Reference

## Globals (pre-imported from `kame`)

```python
Root()                    # Root measurement node
Snapshot(node)            # Immutable read view of a subtree
Transaction(node)         # Write context (optimistic, auto-retry)
sleep(sec)                # KAME-aware sleep (use instead of time.sleep)
```

## Node Navigation

```python
root = Root()
drivers = root["Drivers"]        # Child by name (returns XNode)
driver = drivers["DMM1"]         # Nested access
child = node[0]                  # Child by index
node.getName()                   # Internal name (str)
node.getLabel()                  # UI label (str)
node.getTypename()               # Type name without 'X' prefix
```

## Reading Values (Snapshot)

```python
shot = Snapshot(node)             # Take snapshot of subtree

# Access payload
payload = shot[node]              # Node's payload object

# Convert payload to Python types
float(shot[double_node])          # XDoubleNode -> float
int(shot[int_node])               # XIntNode/XUIntNode -> int
bool(shot[bool_node])             # XBoolNode -> bool
str(shot[value_node])             # Any XValueNodeBase -> str

# List children
children = shot.list(node)        # list[XNode]
for child in shot.list(node):
    print(child.getName())

shot.size(node)                   # Number of children
len(shot)                         # Same as shot.size()
```

## Writing Values

```python
# Simple assignment (auto-transactional)
node["ChildName"] = 3.14         # float, int, bool, or str

# Explicit transaction
for tr in Transaction(node):
    tr[node] = 42                 # Set value
    break                         # Commit

# Transaction with commit control
tr = Transaction(node)
tr[node["SetPoint"]] = 300.0
tr.commit()                       # Returns final Snapshot

# Combo node
tr[combo_node].add("item")       # Add item to combo
tr[combo_node].str("value")      # Set by string

# Touchable node (trigger action)
tr[touchable_node].touch()
```

## Driver Patterns

```python
# List all drivers
drivers = Root()["Drivers"]
shot = Snapshot(drivers)
for d in shot.list(drivers):
    print(d.getName(), d.getTypename())

# Read driver record timestamps
shot = Snapshot(driver)
payload = shot[driver]
payload.time()                    # When phenomenon occurred
payload.timeAwared()              # When visible to operator

# Read driver channel values
voltage = shot[driver].value(0)   # Channel 0

# Access driver child nodes (settings, readings)
shot = Snapshot(driver)
for child in shot.list(driver):
    print(child.getName(), "=", str(shot[child]))
```

## Scalar Entries

```python
entries = Root()["ScalarEntries"]
shot = Snapshot(entries)
for e in shot.list(entries):
    print(e.getName(), "=", str(shot[e]))
```

## Common Recipes

### Read temperature from LakeShore
```python
tc = Root()["Drivers"]["TempControl"]
shot = Snapshot(tc)
for ch in shot.list(tc):
    if "Ch." in ch.getName():
        print(ch.getName(), float(shot[ch]))
```

### Set a parameter and wait
```python
node = Root()["Drivers"]["DCSource"]["Value"]
node["Value"] = "1.0"
sleep(2)
```

### Dump entire driver tree
```python
def dump(node, shot, indent=0):
    name = node.getName()
    try:
        val = str(shot[node])
        print("  " * indent + f"{name} = {val}")
    except Exception:
        print("  " * indent + f"{name}")
    for child in shot.list(node):
        dump(child, shot, indent + 1)

driver = Root()["Drivers"]["DMM1"]
shot = Snapshot(driver)
dump(driver, shot)
```

### Sweep and measure
```python
import time
results = []
source = Root()["Drivers"]["DCSource"]
dmm = Root()["Drivers"]["DMM1"]
for v in [0.1, 0.2, 0.5, 1.0]:
    source["Value"] = str(v)
    sleep(1)
    shot = Snapshot(dmm)
    reading = float(shot[dmm].value(0))
    results.append((v, reading))
    print(f"V={v}, Reading={reading}")
```
