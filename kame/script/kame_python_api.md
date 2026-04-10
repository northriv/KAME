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

To reach `to_png()`, navigate: X2DImage → Graph child → Plots → ImagePlot.

```python
# Full path example for a camera LiveImage:
# Root()["Drivers"]["JAI"]["LiveImage"]        → X2DImage (XGraphNToolBox)
#   ["LiveImage"]                               → XGraph
#     ["Plots"]["ImagePlot"]                    → X2DImagePlot (has to_png())

imgplot = Root()["Drivers"]["JAI"]["LiveImage"]["LiveImage"]["Plots"]["ImagePlot"]
shot = Snapshot(imgplot)
png_bytes = shot[imgplot].to_png()  # QImage as PNG bytes (or None)

# Image dimensions (pixels)
w = shot[imgplot].imageWidth()
h = shot[imgplot].imageHeight()

# Display via IPython (returned as image through MCP)
from IPython.display import display, Image
display(Image(data=png_bytes, format='png'))

# Or load into matplotlib for further analysis
import matplotlib.pyplot as plt, io
img = plt.imread(io.BytesIO(png_bytes))
plt.imshow(img)
plt.gcf()
```

## 2D Math Tools

X2DImage nodes have `Graph2DMathToolList` children (one per channel, e.g. `CH0`, `CH1`).
Use `createByTypename()` to add math tools; `release()` to remove them.

### Available types

| Type name | Description |
|---|---|
| `Graph2DMathToolAverage` | Average pixel value in ROI |
| `Graph2DMathToolSum` | Sum of pixel values in ROI |

### Creating and configuring

ROI coordinates use inclusive pixel endpoints: `FirstX/FirstY` (top-left) and
`LastX/LastY` (bottom-right, inclusive). The ROI width in pixels is `LastX - FirstX + 1`.

```python
ch0 = Root()["Drivers"]["JAI"]["LiveImage"]["CH0"]
dc = ch0.dynamic_cast()  # needed for createByTypename

# Create — returns the new tool node
tool = dc.createByTypename("Graph2DMathToolAverage", "MyAvg")

# Set ROI in pixel coordinates (inclusive endpoints)
tool["FirstX"] = "100"
tool["FirstY"] = "100"
tool["LastX"] = "500"
tool["LastY"] = "500"

# Read the scalar entry value
val_node = tool["CH0-MyAvg"]["Value"]  # name = "{channel}-{toolname}"
shot = Snapshot(val_node)
float(shot[val_node])
```

### Mask types

Each tool has a `MaskType` node: Rectangle (default), Ellipse, Arbitrary.

For Arbitrary masks, use `setArbitraryMask()` which atomically sets MaskType
and the mask bitmap in one transaction:

```python
# Mask: uint8 list, 1=included, 0=excluded.
# Dimensions must be (LastX - FirstX + 1) * (LastY - FirstY + 1).
tool.dynamic_cast().setArbitraryMask(mask_bytes)

# Read back mask
shot = Snapshot(tool)
mask = shot[tool].mask()  # list[uint8]
```

**Important:** Clamp ROI to actual image dimensions to avoid display offset.
Use `shot[imgplot].imageWidth()` and `shot[imgplot].imageHeight()` to get bounds.

### Creating a mask from image analysis

Helper function to create a clean bright-area mask with morphological cleanup
and connected component filtering. Requires `scipy` and `PIL`.

```python
import numpy as np
from scipy import ndimage
from PIL import Image as PILImage

def make_bright_mask(imgplot_path="Drivers/JAI/LiveImage",
                     method="median",    # "median", "otsu", or "percentile"
                     threshold_frac=0.5, # for median: fraction between median and p99
                     percentile=90,      # for percentile method
                     min_area=500,       # discard components smaller than this
                     morph_close=5,      # morphological closing kernel size (0=skip)
                     morph_open=3):      # morphological opening kernel size (0=skip)
    """Analyze image and return (mask_bytes, fx, fy, lx, ly, img_w, img_h).
    mask_bytes is a flat list of uint8 with dimensions (lx-fx+1) * (ly-fy+1)."""
    import io, matplotlib.pyplot as plt

    # Get image
    parts = imgplot_path.strip("/").split("/")
    node = Root()
    for p in parts:
        node = node[p]
    plot = node["LiveImage"]["Plots"]["ImagePlot"]  # X2DImagePlot
    shot = Snapshot(plot)
    img_w, img_h = shot[plot].imageWidth(), shot[plot].imageHeight()
    png = shot[plot].to_png()
    img = plt.imread(io.BytesIO(png))
    gray = np.mean(img[:,:,:3], axis=2) if img.ndim == 3 else img

    # Extract plot area (skip axes margins in rendered image)
    h, w = gray.shape
    px_x0, px_x1 = int(0.02 * w), int(0.98 * w)
    px_y0, px_y1 = int(0.055 * h), int(0.945 * h)
    plot_area = gray[px_y0:px_y1, px_x0:px_x1]
    ph, pw = plot_area.shape

    # Threshold
    if method == "otsu":
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(plot_area)
    elif method == "percentile":
        thresh = np.percentile(plot_area, percentile)
    else:  # median
        med = np.median(plot_area)
        p99 = np.percentile(plot_area, 99)
        thresh = med + threshold_frac * (p99 - med)

    binary = (plot_area > thresh).astype(np.uint8)

    # Morphological cleanup
    if morph_close > 0:
        binary = ndimage.binary_closing(binary, structure=np.ones((morph_close, morph_close))).astype(np.uint8)
    if morph_open > 0:
        binary = ndimage.binary_opening(binary, structure=np.ones((morph_open, morph_open))).astype(np.uint8)

    # Keep only the largest connected component
    labeled, n = ndimage.label(binary)
    if n > 1:
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        binary = (labeled == largest).astype(np.uint8)
    # Remove small components
    if min_area > 0:
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            if ndimage.sum(binary, labeled, i) < min_area:
                binary[labeled == i] = 0

    # Map to image pixel coordinates and clamp
    rows, cols = np.where(binary)
    if len(rows) == 0:
        return [], 0, 0, 0, 0, img_w, img_h
    fx = max(0, int((cols.min() / pw) * img_w))
    lx = min(img_w - 1, int((cols.max() / pw) * img_w))
    fy = max(0, int((rows.min() / ph) * img_h))
    ly = min(img_h - 1, int((rows.max() / ph) * img_h))
    roi_w, roi_h = lx - fx + 1, ly - fy + 1

    # Resize mask to ROI dimensions
    mask_resized = np.array(PILImage.fromarray(binary * 255).resize(
        (roi_w, roi_h), PILImage.NEAREST)) > 127
    return mask_resized.astype(np.uint8).flatten().tolist(), fx, fy, lx, ly, img_w, img_h
```

Usage:
```python
mask_bytes, fx, fy, lx, ly, img_w, img_h = make_bright_mask()

ch0 = Root()["Drivers"]["JAI"]["LiveImage"]["CH0"]
dc = ch0.dynamic_cast()
tool = dc.createByTypename("Graph2DMathToolAverage", "BrightAvg")
tool["FirstX"] = str(fx)
tool["FirstY"] = str(fy)
tool["LastX"] = str(lx)
tool["LastY"] = str(ly)
tool.dynamic_cast().setArbitraryMask(mask_bytes)
```

### Removing a tool

```python
ch0 = Root()["Drivers"]["JAI"]["LiveImage"]["CH0"]
dc = ch0.dynamic_cast()
shot = Snapshot(ch0)
children = shot.list(ch0)
dc.release(children[0])  # release by reference from list
```
