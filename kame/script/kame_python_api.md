# KAME Python API Quick Reference

## Globals (pre-imported from `kame`)

```python
Root()                    # Root measurement node (XMeasure)
Snapshot(node)            # Immutable read view of a subtree
Transaction(node)         # Write context (optimistic, auto-retry)
sleep(sec)                # KAME-aware sleep (use instead of time.sleep)
```

## Node Navigation

```python
root = Root()
drivers = root["Drivers"]        # Child by name (returns XNode or None)
driver = drivers["DMM1"]         # Nested access
child = node[0]                  # Child by index (takes fresh Snapshot)
node.getName()                   # Internal name (str)
node.getLabel()                  # UI label (str)
node.getTypename()               # Type name without 'X' prefix
node.dynamic_cast()              # Downcast to most specific Python type
```

**Important:** `node["name"]` returns `None` if the child doesn't exist.
Always check for `None` before using the result.

## Reading Values (Snapshot)

A Snapshot must be taken from the node or an ancestor that contains it.

```python
shot = Snapshot(node)             # Snapshot of node's subtree

# Access payload — shot[x] only works if x is within shot's subtree
payload = shot[node]              # Node's payload object

# Convert payload to Python types (depends on node type)
float(shot[double_node])          # XDoubleNode -> float
int(shot[int_node])               # XIntNode/XUIntNode -> int
bool(shot[bool_node])             # XBoolNode -> bool
str(node)                         # Any XValueNodeBase -> str (takes own snapshot)

# List children — node must be in the snapshot's subtree
children = shot.list(node)        # list[XNode]
for child in shot.list(node):
    print(child.getName())

shot.size(node)                   # Number of children
len(shot)                         # Children of snapshot root
```

## Writing Values

```python
# Simple assignment (auto-transactional, by child name)
node["ChildName"] = 3.14         # float, int, bool, or str

# Set a value node directly
value_node.set("300.0")           # XValueNodeBase.set(str)

# Explicit transaction
for tr in Transaction(node):
    tr[node] = 42                 # Set value
    break                         # Commit

# Transaction with commit control
tr = Transaction(node)
tr[node["SetPoint"]] = 300.0
tr.commit()                       # Returns final Snapshot

# Touchable node (trigger action)
touchable_node.touch()
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

# Read driver channel values (if driver has value() method)
voltage = shot[driver].value(0)   # Channel 0

# Access driver child nodes (settings, readings)
shot = Snapshot(driver)
for child in shot.list(driver):
    name = child.getName()
    try:
        val = str(child)          # Works for XValueNodeBase children
        print(f"{name} = {val}")
    except Exception:
        print(name)
```

## Scalar Entries

Each `XScalarEntry` has child nodes: `Value` (XDoubleNode), `StoredValue`,
`Delta`, `Store`. Read the numeric value via `entry["Value"]`.

```python
entries = Root()["ScalarEntries"]
shot = Snapshot(entries)
for e in shot.list(entries):
    val_node = e["Value"]
    print(e.getName(), "=", float(shot[val_node]))
    # e.driver() returns the owning XDriver
```

## Root Children

The root node (`Root()`) has these children:

| Name | Type | Runtime |
|---|---|---|
| `Drivers` | XDriverList | no |
| `Interfaces` | XInterfaceList | yes |
| `ScalarEntries` | XScalarEntryList | yes |
| `Thermometers` | XThermometerList | no |
| `GraphList` | XGraphList | no |
| `ChartList` | XChartList | yes |
| `CalibratedEntries` | XCalibratedEntryList | no |

## Common Recipes

### List all drivers and their types
```python
drivers = Root()["Drivers"]
shot = Snapshot(drivers)
for d in shot.list(drivers):
    print(d.getName(), d.getTypename())
```

### Read a value node by path
```python
node = Root()["Drivers"]["TempControl"]["Ch.B"]
shot = Snapshot(node)
print(float(shot[node]))
```

### Dump all children of a driver
```python
driver = Root()["Drivers"]["DMM1"]
shot = Snapshot(driver)
for child in shot.list(driver):
    try:
        print(child.getName(), "=", str(child))
    except Exception:
        print(child.getName())
```

### Set a parameter and wait
```python
source = Root()["Drivers"]["DCSource"]
source["Value"] = "1.0"
sleep(2)
```

### Sweep and measure
```python
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

### Read all scalar entry values
```python
entries = Root()["ScalarEntries"]
shot = Snapshot(entries)
for e in shot.list(entries):
    val = float(shot[e["Value"]])
    print(f"{e.getName()} ({e.driver().getName()}): {val}")
```

## Graph Data Access

### XWaveNGraph (1D plots)

```python
shot = Snapshot(wave_graph)
p = shot[wave_graph]
p.colCount()                      # Number of columns
p.rowCount()                      # Number of rows
p.labels()                        # Column labels (list[str])
p.getColumn(0)                    # Column data (list[float])

# Plot with matplotlib
import matplotlib.pyplot as plt
x = p.getColumn(0)
y = p.getColumn(1)
labels = p.labels()
plt.plot(x, y)
plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.gcf()
```

### X2DImagePlot (2D images)

```python
shot = Snapshot(image_plot)
png_bytes = shot[image_plot].to_png()  # QImage as PNG bytes (or None)

# Display via IPython (returned as image through MCP)
from IPython.display import display, Image
display(Image(data=png_bytes, format='png'))

# Or load into matplotlib for further analysis
import matplotlib.pyplot as plt, io
img = plt.imread(io.BytesIO(png_bytes))
plt.imshow(img)
plt.gcf()
```
