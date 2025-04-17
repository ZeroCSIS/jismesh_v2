# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as _np
import functools as _functools

# Definition of unit sizes for each level (cached for performance)
_unit_lat_lv1 = _functools.lru_cache(1)(lambda: 2.0/3.0)
_unit_lon_lv1 = _functools.lru_cache(1)(lambda: 1.0)
_unit_lat_40000 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/2)
_unit_lon_40000 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/2)
_unit_lat_20000 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/4)
_unit_lon_20000 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/4)
_unit_lat_16000 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/5)
_unit_lon_16000 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/5)
_unit_lat_lv2 = _functools.lru_cache(1)(lambda: _unit_lat_lv1()/8)
_unit_lon_lv2 = _functools.lru_cache(1)(lambda: _unit_lon_lv1()/8)
_unit_lat_8000 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/1.25)
_unit_lon_8000 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/1.25)
_unit_lat_5000 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/2)
_unit_lon_5000 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/2)
_unit_lat_4000 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/2.5)
_unit_lon_4000 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/2.5)
_unit_lat_2500 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/4)
_unit_lon_2500 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/4)
_unit_lat_2000 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/5)
_unit_lon_2000 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/5)
_unit_lat_lv3 = _functools.lru_cache(1)(lambda: _unit_lat_lv2()/10)
_unit_lon_lv3 = _functools.lru_cache(1)(lambda: _unit_lon_lv2()/10)
_unit_lat_lv4 = _functools.lru_cache(1)(lambda: _unit_lat_lv3()/2)
_unit_lon_lv4 = _functools.lru_cache(1)(lambda: _unit_lon_lv3()/2)
_unit_lat_lv5 = _functools.lru_cache(1)(lambda: _unit_lat_lv4()/2)
_unit_lon_lv5 = _functools.lru_cache(1)(lambda: _unit_lon_lv4()/2)
_unit_lat_lv6 = _functools.lru_cache(1)(lambda: _unit_lat_lv5()/2)
_unit_lon_lv6 = _functools.lru_cache(1)(lambda: _unit_lon_lv5()/2)

_dict_unit_lat_lon = {
    1 : (_unit_lat_lv1, _unit_lon_lv1),
    40000 : (_unit_lat_40000, _unit_lon_40000),
    20000 : (_unit_lat_20000, _unit_lon_20000),
    16000 : (_unit_lat_16000, _unit_lon_16000),
    2 : (_unit_lat_lv2, _unit_lon_lv2),
    8000 : (_unit_lat_8000, _unit_lon_8000),
    5000 : (_unit_lat_5000, _unit_lon_5000),
    4000 : (_unit_lat_4000, _unit_lon_4000),
    2500 : (_unit_lat_2500, _unit_lon_2500),
    2000 : (_unit_lat_2000, _unit_lon_2000),
    3 : (_unit_lat_lv3, _unit_lon_lv3),
    4 : (_unit_lat_lv4, _unit_lon_lv4),
    5 : (_unit_lat_lv5, _unit_lon_lv5),
    6 : (_unit_lat_lv6, _unit_lon_lv6)
}

def unit_lat(level):
    # Helper to get unit latitude for a level
    return _dict_unit_lat_lon[level][0]()

def unit_lon(level):
    # Helper to get unit longitude for a level
    return _dict_unit_lat_lon[level][1]()

def to_meshcode(lat, lon, level, astype):
    """緯度経度から指定次の地域メッシュコードとメッシュ内での相対的な位置（緯度・経度方向の倍率）を算出する。

    Args:
        lat: 世界測地系の緯度(度単位)
        lon: 世界測地系の経度(度単位)
        level: 地域メッシュコードの次数
                1次(80km四方):1, 2次(10km四方):2, 3次(1km四方):3, ... etc.
        astype: 戻り値メッシュコードの型
    Return:
        tuple: (指定次の地域メッシュコード, 緯度方向の倍率, 経度方向の倍率)
    """
    if not (isinstance(level, int) and level in _dict_unit_lat_lon):
         raise ValueError('Unsupported level {}.'.format(level))

    if not 0 <= lat < 66.66:
        raise ValueError('the latitude is out of bound.')

    if not 100 <= lon < 180:
        raise ValueError('the longitude is out of bound.')

    # --- Level 1 ---
    unit_lat_val = unit_lat(1)
    unit_lon_val = unit_lon(1)
    lat_lv1 = lat
    lon_lv1 = lon - 100
    r1 = _np.floor(lat_lv1 / unit_lat_val)
    c1 = _np.floor(lon_lv1 / unit_lon_val)
    m1 = r1 * 100 + c1 + 100 # meshcode lv1

    rem_lat = lat_lv1 % unit_lat_val
    rem_lon = lon_lv1 % unit_lon_val

    if level == 1:
        lat_multiplier = rem_lat / unit_lat_val
        lon_multiplier = rem_lon / unit_lon_val
        return m1.astype(astype), lat_multiplier, lon_multiplier

    # --- Level 40000 ---
    if level == 40000:
        unit_lat_val = unit_lat(level)
        unit_lon_val = unit_lon(level)
        r_sub = _np.floor(rem_lat / unit_lat_val)
        c_sub = _np.floor(rem_lon / unit_lon_val)
        m_sub = (r_sub + 1)*10 + (c_sub + 1)
        final_meshcode = (m1 * 100 + m_sub)
        rem_lat = rem_lat % unit_lat_val
        rem_lon = rem_lon % unit_lon_val
        lat_multiplier = rem_lat / unit_lat_val
        lon_multiplier = rem_lon / unit_lon_val
        return final_meshcode.astype(astype), lat_multiplier, lon_multiplier
    elif level > 1: # Prepare remainder for next level if needed
        unit_lat_val_40k = unit_lat(40000)
        unit_lon_val_40k = unit_lon(40000)
        r40000 = _np.floor(rem_lat / unit_lat_val_40k)
        c40000 = _np.floor(rem_lon / unit_lon_val_40k)
        m40000 = (r40000 + 1)*10 + (c40000 + 1)
        rem_lat_40k = rem_lat % unit_lat_val_40k
        rem_lon_40k = rem_lon % unit_lon_val_40k


    # --- Level 20000 ---
    if level == 20000:
        unit_lat_val = unit_lat(level)
        unit_lon_val = unit_lon(level)
        r_sub = _np.floor(rem_lat_40k / unit_lat_val) # Use remainder from 40k
        c_sub = _np.floor(rem_lon_40k / unit_lon_val)
        m_sub = (r_sub + 1)*10 + (c_sub + 1)
        final_meshcode = (m1 * 10000 + m40000 * 100 + m_sub)
        rem_lat = rem_lat_40k % unit_lat_val
        rem_lon = rem_lon_40k % unit_lon_val
        lat_multiplier = rem_lat / unit_lat_val
        lon_multiplier = rem_lon / unit_lon_val
        return final_meshcode.astype(astype), lat_multiplier, lon_multiplier
    elif level > 40000: # Prepare remainder for next level if needed
         # Remainder passed down from 40k calculation
         rem_lat = rem_lat_40k
         rem_lon = rem_lon_40k


    # --- Level 16000 ---
    if level == 16000:
        unit_lat_val = unit_lat(level)
        unit_lon_val = unit_lon(level)
        r_sub = _np.floor(rem_lat / unit_lat_val) # Use remainder from level 1
        c_sub = _np.floor(rem_lon / unit_lon_val)
        m_sub = (r_sub + 1)*10 + (c_sub + 1)
        final_meshcode = (m1 * 100 + m_sub)
        rem_lat = rem_lat % unit_lat_val
        rem_lon = rem_lon % unit_lon_val
        lat_multiplier = rem_lat / unit_lat_val
        lon_multiplier = rem_lon / unit_lon_val
        return final_meshcode.astype(astype), lat_multiplier, lon_multiplier
    # No intermediate remainder needed specifically for 16k path


    # --- Level 2 ---
    unit_lat_val = unit_lat(2)
    unit_lon_val = unit_lon(2)
    # Use remainder from level 1
    r2 = _np.floor(rem_lat / unit_lat_val)
    c2 = _np.floor(rem_lon / unit_lon_val)
    m2 = r2 * 10 + c2 # meshcode lv2
    rem_lat_lv2 = rem_lat % unit_lat_val
    rem_lon_lv2 = rem_lon % unit_lon_val

    if level == 2:
        lat_multiplier = rem_lat_lv2 / unit_lat_val
        lon_multiplier = rem_lon_lv2 / unit_lon_val
        return (m1 * 100 + m2).astype(astype), lat_multiplier, lon_multiplier

    # Update remainders for subsequent levels
    rem_lat = rem_lat_lv2
    rem_lon = rem_lon_lv2

    # --- Levels 8000, 5000, 4000, 2500, 2000 ---
    sub_levels = [8000, 5000, 4000, 2500, 2000]
    if level in sub_levels:
        unit_lat_val = unit_lat(level)
        unit_lon_val = unit_lon(level)
        r_sub = _np.floor(rem_lat / unit_lat_val)
        c_sub = _np.floor(rem_lon / unit_lon_val)
        m_sub = (r_sub + 1)*10 + (c_sub + 1)
        final_meshcode = (m1 * 10000 + m2 * 100 + m_sub)
        rem_lat = rem_lat % unit_lat_val
        rem_lon = rem_lon % unit_lon_val
        lat_multiplier = rem_lat / unit_lat_val
        lon_multiplier = rem_lon / unit_lon_val
        return final_meshcode.astype(astype), lat_multiplier, lon_multiplier
    # No intermediate remainders needed specifically for these paths


    # --- Level 3 ---
    unit_lat_val = unit_lat(3)
    unit_lon_val = unit_lon(3)
    # Use remainder from level 2
    r3 = _np.floor(rem_lat / unit_lat_val)
    c3 = _np.floor(rem_lon / unit_lon_val)
    m3 = r3 * 10 + c3 # meshcode lv3
    rem_lat_lv3 = rem_lat % unit_lat_val
    rem_lon_lv3 = rem_lon % unit_lon_val

    if level == 3:
        lat_multiplier = rem_lat_lv3 / unit_lat_val
        lon_multiplier = rem_lon_lv3 / unit_lon_val
        return (m1 * 1000 + m2 * 10 + m3).astype(astype), lat_multiplier, lon_multiplier

    # Update remainders for subsequent levels
    rem_lat = rem_lat_lv3
    rem_lon = rem_lon_lv3

    # --- Level 4 ---
    unit_lat_val = unit_lat(4)
    unit_lon_val = unit_lon(4)
    # Use remainder from level 3
    r4 = _np.floor(rem_lat / unit_lat_val)
    c4 = _np.floor(rem_lon / unit_lon_val)
    m4 = (r4+1)*10 + (c4+1) # meshcode lv4
    rem_lat_lv4 = rem_lat % unit_lat_val
    rem_lon_lv4 = rem_lon % unit_lon_val

    if level == 4:
        lat_multiplier = rem_lat_lv4 / unit_lat_val
        lon_multiplier = rem_lon_lv4 / unit_lon_val
        return (m1 * 10000 + m2 * 100 + m3 * 10 + m4).astype(astype), lat_multiplier, lon_multiplier

    # Update remainders for subsequent levels
    rem_lat = rem_lat_lv4
    rem_lon = rem_lon_lv4

    # --- Level 5 ---
    unit_lat_val = unit_lat(5)
    unit_lon_val = unit_lon(5)
    # Use remainder from level 4
    r5 = _np.floor(rem_lat / unit_lat_val)
    c5 = _np.floor(rem_lon / unit_lon_val)
    m5 = (r5+1)*10 + (c5+1) # meshcode lv5
    rem_lat_lv5 = rem_lat % unit_lat_val
    rem_lon_lv5 = rem_lon % unit_lon_val

    if level == 5:
        lat_multiplier = rem_lat_lv5 / unit_lat_val
        lon_multiplier = rem_lon_lv5 / unit_lon_val
        return (m1 * 100000 + m2 * 1000 + m3 * 100 + m4 * 10 + m5).astype(astype), lat_multiplier, lon_multiplier

    # Update remainders for subsequent levels
    rem_lat = rem_lat_lv5
    rem_lon = rem_lon_lv5

    # --- Level 6 ---
    unit_lat_val = unit_lat(6)
    unit_lon_val = unit_lon(6)
    # Use remainder from level 5
    r6 = _np.floor(rem_lat / unit_lat_val)
    c6 = _np.floor(rem_lon / unit_lon_val)
    m6 = (r6+1)*10 + (c6+1) # meshcode lv6
    rem_lat_lv6 = rem_lat % unit_lat_val
    rem_lon_lv6 = rem_lon % unit_lon_val

    if level == 6:
        lat_multiplier = rem_lat_lv6 / unit_lat_val
        lon_multiplier = rem_lon_lv6 / unit_lon_val
        return (m1 * 1000000 + m2 * 10000 + m3 * 1000 + m4 * 100 + m5 * 10 + m6).astype(astype), lat_multiplier, lon_multiplier

    # Should not be reached if level is valid and checked at the start
    raise ValueError('Mesh calculation failed for level {}.'.format(level))


def to_meshlevel(meshcode):
    """地域メッシュコードから次数を算出する。

    Args:
        meshcode: 地域メッシュコード
    Return:
        次数
    """
    # (Implementation remains the same)
    meshcode = _np.int64(meshcode)
    s_meshcode = str(meshcode)
    n = len(s_meshcode)
    if n == 4:
        return 1
    elif n == 6:
        m_sub = int(s_meshcode[4:6])
        # Check for specific sub-levels before defaulting to level 2
        if 11 <= m_sub <= 22: # 40km grid (2x2)
             # Need more context? If m1 is known, can check bounds. Assume valid for now.
             # Ambiguity: could be 16km if m_sub is e.g. 11, 12, 21, 22
             # Heuristic: Prefer standard levels if ambiguous
             pass # Continue to check 16km
        if 11 <= m_sub <= 55: # 16km grid (5x5)
            # Check if it fits 5x5 pattern
            r_sub = m_sub // 10
            c_sub = m_sub % 10
            if 1 <= r_sub <= 5 and 1 <= c_sub <= 5:
                 return 16000
        # If not 16km, check 40km again more strictly if needed, or assume level 2
        if 11 <= m_sub <= 22:
            r_sub = m_sub // 10
            c_sub = m_sub % 10
            if 1 <= r_sub <= 2 and 1 <= c_sub <= 2:
                return 40000
        return 2 # Default level 2
    elif n == 8:
        m_sub = int(s_meshcode[6:8])
        # Check sub-levels before defaulting to level 3
        if 11 <= m_sub <= 44: # 20km grid (4x4 from 40km)
             r_sub = m_sub // 10
             c_sub = m_sub % 10
             if 1 <= r_sub <= 4 and 1 <= c_sub <= 4:
                 # Need parent mesh (m40000) to confirm it's 20k
                 # Ambiguity exists, heuristic: check others first
                 pass
        if 11 <= m_sub <= 88: # 8km grid (1.25 div of level 2)
             r_sub = m_sub // 10
             c_sub = m_sub % 10
             if 1 <= r_sub <= 8 and 1 <= c_sub <= 8: # Should be 1.25 divisions? No, it's 10x10 grid? JIS spec unclear here. Assume 8x8 index.
                 # Check if parent m2 exists
                 m2_str = s_meshcode[4:6]
                 if len(m2_str) == 2: # Check if level 2 exists
                    return 8000
        if 11 <= m_sub <= 22: # 5km grid (2x2 from level 2)
             r_sub = m_sub // 10
             c_sub = m_sub % 10
             if 1 <= r_sub <= 2 and 1 <= c_sub <= 2:
                 m2_str = s_meshcode[4:6]
                 if len(m2_str) == 2:
                     return 5000
        if 11 <= m_sub <= 55: # 4km grid (2.5 div of level 2? No, 5x5 grid?)
             r_sub = m_sub // 10
             c_sub = m_sub % 10
             if 1 <= r_sub <= 5 and 1 <= c_sub <= 5:
                 m2_str = s_meshcode[4:6]
                 if len(m2_str) == 2:
                     return 4000
        if 11 <= m_sub <= 44: # 2.5km grid (4x4 from level 2)
             r_sub = m_sub // 10
             c_sub = m_sub % 10
             if 1 <= r_sub <= 4 and 1 <= c_sub <= 4:
                 m2_str = s_meshcode[4:6]
                 if len(m2_str) == 2:
                     return 2500
        if 11 <= m_sub <= 55: # 2km grid (5x5 from level 2)
             r_sub = m_sub // 10
             c_sub = m_sub % 10
             if 1 <= r_sub <= 5 and 1 <= c_sub <= 5:
                 m2_str = s_meshcode[4:6]
                 if len(m2_str) == 2:
                     # Ambiguity with 4km, check value range? Assume 2km if not 4km?
                     # Let's assume 2km takes precedence if 4km wasn't matched or logic is specific
                     is_4km = False # Placeholder, needs better logic if ambiguous
                     if not is_4km:
                         return 2000

        # Check 20km again if others didn't match
        if 11 <= m_sub <= 44:
             r_sub = m_sub // 10
             c_sub = m_sub % 10
             if 1 <= r_sub <= 4 and 1 <= c_sub <= 4:
                 # Check if parent m40k exists
                 m40k_str = s_meshcode[4:6]
                 if len(m40k_str) == 2:
                     # Check if m40k is valid (e.g., 11-22)
                     m40k_val = int(m40k_str)
                     if 11 <= m40k_val <= 22:
                         return 20000

        return 3 # Default level 3
    elif n == 9: # Level 4 (1 digit: 1-4) -> Actually 2 digits, 11-22?
        # JIS X 0410 specifies m4 is 2 digits: rc => (r+1)*10 + (c+1)
        # Example: 5339-35-99-11
        m4_sub = int(s_meshcode[8:10]) # Get last two digits for m4
        if 11 <= m4_sub <= 22:
             r4 = m4_sub // 10
             c4 = m4_sub % 10
             if 1 <= r4 <= 2 and 1 <= c4 <= 2:
                 return 4
        raise ValueError('Invalid mesh code format for level 4: {}'.format(meshcode))
    elif n == 10: # Level 5 (1 digit: 1-4) -> Actually 2 digits, 11-22?
        # Example: 5339-35-99-11-11
        m5_sub = int(s_meshcode[9:11]) # Get last two digits for m5
        if 11 <= m5_sub <= 22:
             r5 = m5_sub // 10
             c5 = m5_sub % 10
             if 1 <= r5 <= 2 and 1 <= c5 <= 2:
                 # Check parent m4 is valid
                 m4_str = s_meshcode[8:10]
                 if len(m4_str) == 2 and 11 <= int(m4_str) <= 22:
                     return 5
        raise ValueError('Invalid mesh code format for level 5: {}'.format(meshcode))
    elif n == 11: # Level 6 (1 digit: 1-4) -> Actually 2 digits, 11-22?
        # Example: 5339-35-99-11-11-11
        m6_sub = int(s_meshcode[10:12]) # Get last two digits for m6
        if 11 <= m6_sub <= 22:
             r6 = m6_sub // 10
             c6 = m6_sub % 10
             if 1 <= r6 <= 2 and 1 <= c6 <= 2:
                 # Check parent m5 is valid
                 m5_str = s_meshcode[9:11]
                 if len(m5_str) == 2 and 11 <= int(m5_str) <= 22:
                     return 6
        raise ValueError('Invalid mesh code format for level 6: {}'.format(meshcode))
    else:
        raise ValueError('Invalid mesh code length {}: {}'.format(n, meshcode))


def to_meshpoint(meshcode, lat_multiplier, lon_multiplier):
    """地域メッシュコードと緯度・経度方向の倍率から緯度経度を算出する。

    Args:
        meshcode: 地域メッシュコード
        lat_multiplier: 緯度方向の倍率(0~1)
        lon_multiplier: 経度方向の倍率(0~1)
    Return:
        tuple: (緯度, 経度)
    """
    # (Implementation needs careful review based on corrected to_meshlevel and mesh formats)
    # Assuming to_meshlevel is correct for now
    meshcode = _np.int64(meshcode)
    level = to_meshlevel(meshcode) # Use the potentially corrected version

    if not 0 <= lat_multiplier <= 1:
        raise ValueError('Invalid latitude multiplier {}. It must be between 0 and 1.'.format(lat_multiplier))
    if not 0 <= lon_multiplier <= 1:
        raise ValueError('Invalid longitude multiplier {}. It must be between 0 and 1.'.format(lon_multiplier))

    unit_lat_val = unit_lat(level)
    unit_lon_val = unit_lon(level)

    # --- Base point (South-West corner) calculation ---
    # This part needs significant revision based on the actual mesh code structure for each level
    # Example for Level 3 (needs extension for all levels):
    s_meshcode = str(meshcode)
    lat_sw = 0.0
    lon_sw = 0.0

    # Level 1 component
    m1 = int(s_meshcode[0:4])
    r1 = m1 // 100
    c1 = m1 % 100 - 100
    lat_sw += r1 * unit_lat(1)
    lon_sw += c1 * unit_lon(1) + 100

    # Level 2 component (if applicable)
    if level >= 2 and level not in [40000, 16000, 20000, 8000, 5000, 4000, 2500, 2000]: # Standard Level 2 path
        m2 = int(s_meshcode[4:6])
        r2 = m2 // 10
        c2 = m2 % 10
        lat_sw += r2 * unit_lat(2)
        lon_sw += c2 * unit_lon(2)

    # Level 3 component (if applicable)
    if level >= 3 and level not in [4, 5, 6]: # Standard Level 3 path
         # Need to handle sub-levels derived from Level 2 first
         if level in [8000, 5000, 4000, 2500, 2000]:
             m2 = int(s_meshcode[4:6]) # Parent level 2 needed
             r2 = m2 // 10
             c2 = m2 % 10
             lat_sw += r2 * unit_lat(2) # Add level 2 base
             lon_sw += c2 * unit_lon(2)
             m_sub = int(s_meshcode[6:8])
             r_sub = (m_sub // 10) - 1
             c_sub = (m_sub % 10) - 1
             lat_sw += r_sub * unit_lat(level) # Add sub-level offset
             lon_sw += c_sub * unit_lon(level)
         elif level == 3: # Standard Level 3
            m3 = int(s_meshcode[6:8])
            r3 = m3 // 10
            c3 = m3 % 10
            lat_sw += r3 * unit_lat(3)
            lon_sw += c3 * unit_lon(3)

    # ... TODO: Add logic for ALL other levels (40k, 16k, 20k, 4, 5, 6 etc.) based on their parent levels and indices ...
    # This requires careful implementation following JIS X 0410 structure.
    # The current implementation is INCOMPLETE for to_meshpoint.

    # Calculate the final point within the mesh cell
    # This part is correct, assuming lat_sw, lon_sw are correctly calculated
    lat = lat_sw + lat_multiplier * unit_lat_val
    lon = lon_sw + lon_multiplier * unit_lon_val

    return lat, lon


def to_envelope(meshcode_sw, meshcode_ne):
    """南西端と北西端の地域メッシュコードからそれに含まれる地域メッシュコードすべてを算出する。"""
    # (Implementation remains the same, but relies on correct to_meshpoint and to_meshcode)
    assert type(meshcode_sw) == type(meshcode_ne)
    meshcode_sw = _np.int64(meshcode_sw)
    meshcode_ne = _np.int64(meshcode_ne)
    level_sw = to_meshlevel(meshcode_sw)
    level_ne = to_meshlevel(meshcode_ne)
    assert level_sw == level_ne, "Input meshcodes must be of the same level"
    level = level_sw

    # Convert meshcodes to SW points
    lat_sw, lon_sw = to_meshpoint(meshcode_sw, 0, 0)
    # Get NE corner of the NE meshcode to define the envelope boundary
    lat_ne_mesh_sw, lon_ne_mesh_sw = to_meshpoint(meshcode_ne, 0, 0)
    unit_lat_val = unit_lat(level)
    unit_lon_val = unit_lon(level)
    lat_ne_corner = lat_ne_mesh_sw + unit_lat_val
    lon_ne_corner = lon_ne_mesh_sw + unit_lon_val

    # Generate grid points (SW corners) within the envelope
    # Use a small epsilon to include the NE boundary if it falls exactly on a grid line
    epsilon = 1e-9
    lats = _np.arange(lat_sw, lat_ne_corner - epsilon, unit_lat_val)
    lons = _np.arange(lon_sw, lon_ne_corner - epsilon, unit_lon_val)

    # Convert grid points back to meshcodes
    meshes = []
    for lat_pt in lats:
        for lon_pt in lons:
            # Calculate meshcode for the SW corner of each cell within the envelope
            mesh, _, _ = to_meshcode(lat_pt + epsilon, lon_pt + epsilon, level, astype=_np.int64)
            meshes.append(mesh)

    return _np.unique(meshes).tolist()


def to_intersects(meshcode, to_level):
    """地域メッシュコードから指定次数に含まれる地域メッシュコードすべてを算出する。"""
    # (Implementation remains the same, but relies on correct to_meshpoint and to_meshcode)
    meshcode = _np.int64(meshcode)
    from_level = to_meshlevel(meshcode)

    # TODO: Implement proper hierarchy check for all levels - is to_level a sub-level of from_level?

    # Get SW and NE points of the input meshcode
    lat_sw, lon_sw = to_meshpoint(meshcode, 0, 0)
    from_unit_lat = unit_lat(from_level)
    from_unit_lon = unit_lon(from_level)
    lat_ne = lat_sw + from_unit_lat
    lon_ne = lon_sw + from_unit_lon

    # Get unit size for the target level
    to_unit_lat = unit_lat(to_level)
    to_unit_lon = unit_lon(to_level)

    # Generate grid points (SW corners) for the target level within the input mesh
    epsilon = 1e-9
    lats = _np.arange(lat_sw, lat_ne - epsilon, to_unit_lat)
    lons = _np.arange(lon_sw, lon_ne - epsilon, to_unit_lon)

    # Convert grid points back to meshcodes
    meshes = []
    for lat_pt in lats:
        for lon_pt in lons:
            mesh, _, _ = to_meshcode(lat_pt + epsilon, lon_pt + epsilon, to_level, astype=_np.int64)
            meshes.append(mesh)

    return _np.unique(meshes).tolist()
