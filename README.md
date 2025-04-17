A fork of hni14/jismesh, updated to return relative latitude/longitude position (multipliers) along with the mesh code from the to_meshcode function.
Here is an example case of usage:

```
import jismesh.utils as ju
import numpy as np

# Scalar example
meshcode, lat_mult, lon_mult = ju.to_meshcode(35.658581, 139.745433, 3)
print(f"Meshcode: {meshcode}, Lat Multiplier: {lat_mult:.3f}, Lon Multiplier: {lon_mult:.3f}")
# Expected Output (approx): Meshcode: 53393599, Lat Multiplier: 0.903, Lon Multiplier: 0.443

# Vector example
lats = np.array([35.658581, 35.681236])
lons = np.array([139.745433, 139.767125])
meshcodes, lat_mults, lon_mults = ju.to_meshcode(lats, lons, 3)
print(f"Meshcodes: {meshcodes}")
print(f"Lat Multipliers: {np.round(lat_mults, 3)}")
print(f"Lon Multipliers: {np.round(lon_mults, 3)}")
# Expected Output (approx):
# Meshcodes: [53393599 53394681]
# Lat Multipliers: [0.903 0.748]
# Lon Multipliers: [0.443 0.055]
```
