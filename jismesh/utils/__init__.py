from __future__ import absolute_import

import numpy as _np
from . import _vector
from . import _scalar

def unit_lat(level):
    """
    指定されたメッシュレベルの単位緯度（度単位）を取得します。

    Args:
        level: 地域メッシュコードの次数 (scalar or numpy array)

    Returns:
        float or numpy array: 単位緯度
    """
    if _np.isscalar(level):
        unit_lat_func = _scalar.unit_lat
    else:
        unit_lat_func = _vector.unit_lat

    return unit_lat_func(level)

def unit_lon(level):
    """
    指定されたメッシュレベルの単位経度（度単位）を取得します。

    Args:
        level: 地域メッシュコードの次数 (scalar or numpy array)

    Returns:
        float or numpy array: 単位経度
    """
    if _np.isscalar(level):
        unit_lon_func = _scalar.unit_lon
    else:
        unit_lon_func = _vector.unit_lon

    return unit_lon_func(level)

def to_meshcode(lat, lon, level, astype=_np.int64):
    """
    緯度経度から指定次の地域メッシュコードとメッシュ内での相対的な位置（緯度・経度方向の倍率）を算出する。

    Args:
        lat: 世界測地系の緯度(度単位) (scalar or numpy array)
        lon: 世界測地系の経度(度単位) (scalar or numpy array)
        level: 地域メッシュコードの次数 (scalar or numpy array)
                例: 1次(80km):1, 2次(10km):2, 3次(1km):3, 4次(500m):4,
                   5次(250m):5, 6次(125m):6, 5倍(5km):5000, 10倍(1km):3 など
        astype: 戻り値メッシュコードの型 (default: numpy.int64)

    Returns:
        tuple: (地域メッシュコード, 緯度方向の倍率, 経度方向の倍率)
               入力がスカラの場合、(scalar, scalar, scalar) を返します。
               入力が numpy 配列の場合、(numpy array, numpy array, numpy array) を返します。
               緯度・経度方向の倍率は 0.0 から 1.0 の範囲です。
    """
    # Determine if inputs are scalar or array-like
    is_scalar_input = _np.isscalar(lat) and _np.isscalar(lon) and _np.isscalar(level)

    if is_scalar_input:
        # Use scalar implementation directly
        to_meshcode_func = _scalar.to_meshcode
    else:
        # Use vector implementation
        to_meshcode_func = _vector.to_meshcode

    return to_meshcode_func(lat, lon, level, astype)

def to_meshlevel(meshcode):
    """
    地域メッシュコードから次数を算出する。

    Args:
        meshcode: 地域メッシュコード (scalar or numpy array)

    Returns:
        int or numpy array: 次数
    """
    if _np.isscalar(meshcode):
        to_meshlevel_func = _scalar.to_meshlevel
    else:
        to_meshlevel_func = _vector.to_meshlevel

    return to_meshlevel_func(meshcode)

def to_meshpoint(meshcode, lat_multiplier, lon_multiplier):
    """
    地域メッシュコードと緯度・経度方向の倍率から緯度経度（南西端基準）を算出する。

    Args:
        meshcode: 地域メッシュコード (scalar or numpy array)
        lat_multiplier: 緯度方向の倍率(0~1) (scalar or numpy array)
        lon_multiplier: 経度方向の倍率(0~1) (scalar or numpy array)

    Returns:
        tuple: (緯度, 経度)
               入力がスカラの場合、(scalar, scalar) を返します。
               入力が numpy 配列の場合、(numpy array, numpy array) を返します。
    """
    # Determine if inputs are scalar or array-like
    is_scalar_input = _np.isscalar(meshcode) and _np.isscalar(lat_multiplier) and _np.isscalar(lon_multiplier)

    if is_scalar_input:
        to_meshpoint_func = _scalar.to_meshpoint
    else:
        to_meshpoint_func = _vector.to_meshpoint

    return to_meshpoint_func(meshcode, lat_multiplier, lon_multiplier)

# Keep scalar versions for envelope and intersects as they were originally scalar
def to_envelope(meshcode_sw, meshcode_ne):
    """
    南西端と北東端の地域メッシュコードからそれに含まれる地域メッシュコードすべてを算出する。
    入力メッシュコードは同じレベルである必要があります。

    Args:
        meshcode_sw: 南西端の地域メッシュコード (scalar)
        meshcode_ne: 北東端の地域メッシュコード (scalar)

    Returns:
        list: 含まれる地域メッシュコードのリスト (int)
    """
    # Ensure inputs are scalar for this function
    if not (_np.isscalar(meshcode_sw) and _np.isscalar(meshcode_ne)):
        raise TypeError("to_envelope currently only supports scalar inputs.")

    # Use scalar implementation
    to_envelope_func = _scalar.to_envelope
    return to_envelope_func(meshcode_sw, meshcode_ne)

def to_intersects(meshcode, to_level):
    """
    地域メッシュコードから指定次数に含まれる（より細かい）地域メッシュコードすべてを算出する。

    Args:
        meshcode: 地域メッシュコード (scalar)
        to_level: 算出したい地域メッシュコードの次数 (scalar)

    Returns:
        list: 含まれる地域メッシュコードのリスト (int)
    """
    # Ensure inputs are scalar for this function
    if not (_np.isscalar(meshcode) and _np.isscalar(to_level)):
         raise TypeError("to_intersects currently only supports scalar inputs.")

    # Use scalar implementation
    to_intersects_func = _scalar.to_intersects
    return to_intersects_func(meshcode, to_level)
