# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as _np
from ._scalar import (
    _unit_lat_lv1, _unit_lon_lv1, # Import necessary unit functions/values
    _dict_unit_lat_lon,
    unit_lat as _unit_lat_scalar, # Keep scalar versions for lookup
    unit_lon as _unit_lon_scalar,
    to_meshlevel as _to_meshlevel_scalar,
    to_meshpoint as _to_meshpoint_scalar # Keep scalar version for vectorization
)

_supported_levels = _np.array(list(_dict_unit_lat_lon.keys()), dtype=_np.int64)

def unit_lat(level):
    # Vectorized unit_lat lookup
    level = _np.atleast_1d(level).astype(_np.int64)
    if not _np.all(_np.isin(level, _supported_levels)):
        raise ValueError('Unsupported level is specified.')

    # Use vectorize for dynamic lookup based on level values
    v_unit_lat = _np.vectorize(lambda l: _unit_lat_scalar(l))
    result = v_unit_lat(level)

    # Return scalar if input was scalar-like
    if result.size == 1 and _np.isscalar(_np.atleast_1d(level)[0]):
         return _np.asscalar(result)
    return result


def unit_lon(level):
    # Vectorized unit_lon lookup
    level = _np.atleast_1d(level).astype(_np.int64)
    if not _np.all(_np.isin(level, _supported_levels)):
        raise ValueError('Unsupported level is specified.')

    v_unit_lon = _np.vectorize(lambda l: _unit_lon_scalar(l))
    result = v_unit_lon(level)

    if result.size == 1 and _np.isscalar(_np.atleast_1d(level)[0]):
         return _np.asscalar(result)
    return result


def to_meshcode(lat, lon, level, astype):
    """緯度経度から指定次の地域メッシュコードとメッシュ内での相対的な位置（緯度・経度方向の倍率）を算出する。(ベクトル版)

    Args:
        lat: 世界測地系の緯度(度単位) (numpy array)
        lon: 世界測地系の経度(度単位) (numpy array)
        level: 地域メッシュコードの次数 (numpy array or scalar)
        astype: 戻り値メッシュコードの型
    Return:
        tuple: (指定次の地域メッシュコード (numpy array), 緯度方向の倍率 (numpy array), 経度方向の倍率 (numpy array))
    """
    lat = _np.atleast_1d(lat).astype(_np.float64)
    lon = _np.atleast_1d(lon).astype(_np.float64)
    level_in = _np.atleast_1d(level).astype(_np.int64) # Use level_in to avoid shadowing

    # Input validation
    if not _np.all((0 <= lat) & (lat < 66.66)):
        raise ValueError('Input latitude(s) are out of bound [0, 66.66).')
    if not _np.all((100 <= lon) & (lon < 180)):
        raise ValueError('Input longitude(s) are out of bound [100, 180).')
    if not _np.all(_np.isin(level_in, _supported_levels)):
        unsupported = level_in[~_np.isin(level_in, _supported_levels)]
        raise ValueError(f'Unsupported level(s) specified: {unsupported}.')

    # Broadcast arrays to common shape
    try:
        b_lat, b_lon, b_level = _np.broadcast_arrays(lat, lon, level_in)
    except ValueError as e:
        raise ValueError(f"Input arrays lat, lon, level could not be broadcast together. Shapes: {lat.shape}, {lon.shape}, {level_in.shape}. Error: {e}")

    # Initialize result arrays
    meshcode = _np.zeros(b_lat.shape, dtype=astype)
    lat_multiplier = _np.zeros(b_lat.shape, dtype=_np.float64)
    lon_multiplier = _np.zeros(b_lat.shape, dtype=_np.float64)

    # --- Level 1 ---
    unit_lat_lv1_val = _unit_lat_scalar(1)
    unit_lon_lv1_val = _unit_lon_scalar(1)
    lat_lv1 = b_lat
    lon_lv1 = b_lon - 100
    r1 = _np.floor(lat_lv1 / unit_lat_lv1_val)
    c1 = _np.floor(lon_lv1 / unit_lon_lv1_val)
    m1 = r1 * 100 + c1 + 100 # meshcode lv1

    rem_lat = lat_lv1 % unit_lat_lv1_val
    rem_lon = lon_lv1 % unit_lon_lv1_val

    mask = (b_level == 1)
    if _np.any(mask):
        meshcode[mask] = m1[mask].astype(astype)
        lat_multiplier[mask] = rem_lat[mask] / unit_lat_lv1_val
        lon_multiplier[mask] = rem_lon[mask] / unit_lon_lv1_val

    # --- Process other levels ---
    # It's complex to vectorize the entire sequential logic directly.
    # Instead, iterate through unique levels present in the input `b_level` > 1.
    unique_levels = _np.unique(b_level[b_level > 1])

    for current_level in unique_levels:
        mask = (b_level == current_level)
        if not _np.any(mask): continue # Skip if this level isn't requested

        # Get unit sizes for the current level
        unit_lat_val = _unit_lat_scalar(current_level)
        unit_lon_val = _unit_lon_scalar(current_level)

        # Apply scalar logic within the masked elements
        # This is inefficient but avoids full vectorization complexity.
        # A more optimized approach might use np.select or pre-calculate all mesh parts.

        # Re-calculate based on scalar logic for elements matching current_level
        # Note: This recalculates intermediate steps for each level group.
        lat_masked = b_lat[mask]
        lon_masked = b_lon[mask]

        # --- Re-run scalar logic up to current_level ---
        # Level 1 base (already calculated as m1[mask])
        m1_masked = m1[mask]
        rem_lat_masked = rem_lat[mask]
        rem_lon_masked = rem_lon[mask]

        # Intermediate levels needed for current_level
        m_parts = {'m1': m1_masked}
        current_rem_lat = rem_lat_masked
        current_rem_lon = rem_lon_masked

        # Calculate intermediate mesh parts and remainders sequentially
        # Level 40k needed for 20k
        if current_level >= 20000 or current_level in [8000, 5000, 4000, 2500, 2000, 3, 4, 5, 6]: # Levels needing 40k base or its remainder
            ul = _unit_lat_scalar(40000)
            uo = _unit_lon_scalar(40000)
            r = _np.floor(current_rem_lat / ul)
            c = _np.floor(current_rem_lon / uo)
            m_parts['m40k'] = (r + 1)*10 + (c + 1)
            rem_lat_40k = current_rem_lat % ul
            rem_lon_40k = current_rem_lon % uo
            if current_level > 40000: # Pass remainder down
                 current_rem_lat = rem_lat_40k
                 current_rem_lon = rem_lon_40k

        # Level 2 needed for many sub-levels
        if current_level >= 2 or current_level in [8000, 5000, 4000, 2500, 2000, 3, 4, 5, 6]:
             ul = _unit_lat_scalar(2)
             uo = _unit_lon_scalar(2)
             # Use remainder from Level 1 (rem_lat_masked, rem_lon_masked)
             r = _np.floor(rem_lat_masked / ul)
             c = _np.floor(rem_lon_masked / uo)
             m_parts['m2'] = r * 10 + c
             rem_lat_lv2 = rem_lat_masked % ul
             rem_lon_lv2 = rem_lon_masked % uo
             if current_level > 2: # Pass remainder down
                 current_rem_lat = rem_lat_lv2
                 current_rem_lon = rem_lon_lv2

        # Level 3 needed for 4, 5, 6
        if current_level >= 3:
             ul = _unit_lat_scalar(3)
             uo = _unit_lon_scalar(3)
             r = _np.floor(current_rem_lat / ul) # Uses rem_lat_lv2
             c = _np.floor(current_rem_lon / uo) # Uses rem_lon_lv2
             m_parts['m3'] = r * 10 + c
             rem_lat_lv3 = current_rem_lat % ul
             rem_lon_lv3 = current_rem_lon % uo
             if current_level > 3:
                 current_rem_lat = rem_lat_lv3
                 current_rem_lon = rem_lon_lv3

        # Level 4 needed for 5, 6
        if current_level >= 4:
             ul = _unit_lat_scalar(4)
             uo = _unit_lon_scalar(4)
             r = _np.floor(current_rem_lat / ul) # Uses rem_lat_lv3
             c = _np.floor(current_rem_lon / uo) # Uses rem_lon_lv3
             m_parts['m4'] = (r + 1)*10 + (c + 1)
             rem_lat_lv4 = current_rem_lat % ul
             rem_lon_lv4 = current_rem_lon % uo
             if current_level > 4:
                 current_rem_lat = rem_lat_lv4
                 current_rem_lon = rem_lon_lv4

        # Level 5 needed for 6
        if current_level >= 5:
             ul = _unit_lat_scalar(5)
             uo = _unit_lon_scalar(5)
             r = _np.floor(current_rem_lat / ul) # Uses rem_lat_lv4
             c = _np.floor(current_rem_lon / uo) # Uses rem_lon_lv4
             m_parts['m5'] = (r + 1)*10 + (c + 1)
             rem_lat_lv5 = current_rem_lat % ul
             rem_lon_lv5 = current_rem_lon % uo
             if current_level > 5:
                 current_rem_lat = rem_lat_lv5
                 current_rem_lon = rem_lon_lv5

        # --- Calculate final meshcode and multipliers for current_level ---
        final_meshcode_masked = _np.zeros(lat_masked.shape, dtype=astype)
        final_rem_lat = _np.zeros(lat_masked.shape, dtype=_np.float64)
        final_rem_lon = _np.zeros(lat_masked.shape, dtype=_np.float64)

        # Combine mesh parts based on the current_level
        if current_level == 40000:
            final_meshcode_masked = m_parts['m1'] * 100 + m_parts['m40k']
            final_rem_lat = rem_lat_40k # Remainder from its own calculation
            final_rem_lon = rem_lon_40k
        elif current_level == 20000:
            final_meshcode_masked = m_parts['m1'] * 10000 + m_parts['m40k'] * 100 + m_parts['m20k'] # m20k needs calc
            ul = _unit_lat_scalar(20000); uo = _unit_lon_scalar(20000)
            r = _np.floor(rem_lat_40k / ul); c = _np.floor(rem_lon_40k / uo)
            m_parts['m20k'] = (r + 1)*10 + (c + 1)
            final_meshcode_masked = m_parts['m1'] * 10000 + m_parts['m40k'] * 100 + m_parts['m20k']
            final_rem_lat = rem_lat_40k % ul
            final_rem_lon = rem_lon_40k % uo
        elif current_level == 16000:
            ul = _unit_lat_scalar(16000); uo = _unit_lon_scalar(16000)
            r = _np.floor(rem_lat_masked / ul); c = _np.floor(rem_lon_masked / uo) # Uses L1 remainder
            m_parts['m16k'] = (r + 1)*10 + (c + 1)
            final_meshcode_masked = m_parts['m1'] * 100 + m_parts['m16k']
            final_rem_lat = rem_lat_masked % ul
            final_rem_lon = rem_lon_masked % uo
        elif current_level == 2:
            final_meshcode_masked = m_parts['m1'] * 100 + m_parts['m2']
            final_rem_lat = rem_lat_lv2
            final_rem_lon = rem_lon_lv2
        elif current_level in [8000, 5000, 4000, 2500, 2000]:
            ul = _unit_lat_scalar(current_level); uo = _unit_lon_scalar(current_level)
            r = _np.floor(rem_lat_lv2 / ul); c = _np.floor(rem_lon_lv2 / uo) # Uses L2 remainder
            m_sub = (r + 1)*10 + (c + 1)
            final_meshcode_masked = m_parts['m1'] * 10000 + m_parts['m2'] * 100 + m_sub
            final_rem_lat = rem_lat_lv2 % ul
            final_rem_lon = rem_lon_lv2 % uo
        elif current_level == 3:
            final_meshcode_masked = m_parts['m1'] * 1000 + m_parts['m2'] * 10 + m_parts['m3']
            final_rem_lat = rem_lat_lv3
            final_rem_lon = rem_lon_lv3
        elif current_level == 4:
            final_meshcode_masked = m_parts['m1'] * 10000 + m_parts['m2'] * 100 + m_parts['m3'] * 10 + m_parts['m4']
            final_rem_lat = rem_lat_lv4
            final_rem_lon = rem_lon_lv4
        elif current_level == 5:
            final_meshcode_masked = m_parts['m1'] * 100000 + m_parts['m2'] * 1000 + m_parts['m3'] * 100 + m_parts['m4'] * 10 + m_parts['m5']
            final_rem_lat = rem_lat_lv5
            final_rem_lon = rem_lon_lv5
        elif current_level == 6:
            ul = _unit_lat_scalar(6); uo = _unit_lon_scalar(6)
            r = _np.floor(rem_lat_lv5 / ul); c = _np.floor(rem_lon_lv5 / uo) # Uses L5 remainder
            m_parts['m6'] = (r + 1)*10 + (c + 1)
            final_meshcode_masked = m_parts['m1'] * 1000000 + m_parts['m2'] * 10000 + m_parts['m3'] * 1000 + m_parts['m4'] * 100 + m_parts['m5'] * 10 + m_parts['m6']
            final_rem_lat = rem_lat_lv5 % ul
            final_rem_lon = rem_lon_lv5 % uo

        # Assign results to the main arrays using the mask
        meshcode[mask] = final_meshcode_masked.astype(astype)
        # Avoid division by zero if unit size is somehow zero (shouldn't happen)
        lat_multiplier[mask] = _np.divide(final_rem_lat, unit_lat_val, out=_np.zeros_like(final_rem_lat), where=unit_lat_val!=0)
        lon_multiplier[mask] = _np.divide(final_rem_lon, unit_lon_val, out=_np.zeros_like(final_rem_lon), where=unit_lon_val!=0)


    # Return results, potentially squeezing dimensions if input was scalar-like
    # Check original input shapes to decide if squeeze is needed
    if lat.ndim == 0 and lon.ndim == 0 and level_in.ndim == 0:
         return _np.asscalar(meshcode), _np.asscalar(lat_multiplier), _np.asscalar(lon_multiplier)
    else:
         # Return potentially broadcasted shape
         return meshcode, lat_multiplier, lon_multiplier


def to_meshlevel(meshcode):
    """地域メッシュコードから次数を算出する。(ベクトル版)"""
    if _np.isscalar(meshcode):
        return _to_meshlevel_scalar(meshcode)
    else:
        # Vectorize the potentially improved scalar function
        v_meshlevel = _np.vectorize(_to_meshlevel_scalar, otypes=[_np.int64])
        return v_meshlevel(meshcode)

def to_meshpoint(meshcode, lat_multiplier, lon_multiplier):
    """地域メッシュコードと緯度・経度方向の倍率から緯度経度を算出する。(ベクトル版)"""
    if _np.isscalar(meshcode) and _np.isscalar(lat_multiplier) and _np.isscalar(lon_multiplier):
        # Use the scalar version (which needs completion/correction)
        return _to_meshpoint_scalar(meshcode, lat_multiplier, lon_multiplier)
    else:
        # Vectorize the potentially improved scalar function
        v_meshpoint = _np.vectorize(_to_meshpoint_scalar, otypes=[_np.float64, _np.float64])
        # Ensure inputs are broadcastable
        try:
            b_mesh, b_lat_mult, b_lon_mult = _np.broadcast_arrays(meshcode, lat_multiplier, lon_multiplier)
            return v_meshpoint(b_mesh, b_lat_mult, b_lon_mult)
        except ValueError as e:
             raise ValueError(f"Input arrays meshcode, lat_multiplier, lon_multiplier could not be broadcast together. Shapes: {_np.shape(meshcode)}, {_np.shape(lat_multiplier)}, {_np.shape(lon_multiplier)}. Error: {e}")

# Note: to_envelope and to_intersects remain scalar operations.
# If vector versions are needed, they would require significant refactoring.
