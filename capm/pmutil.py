#
# Copyright (C) 2016 The Regents of the University of California
# All Rights Reserved
# Created by the Advanced Highway Maintenance and Construction
# Technology Research Center (AHMCT)
#

import math
import pyproj

from .opgeo2pm import OpGeo2Pm
from .oppm2geo import OpPm2Geo


def get_utm_zone(lon):
    if (lon is None):
        raise ValueError("Invalid parameter: lon is null")
    zone = (int(math.floor((lon + 180.0)/6.0)) % 60) + 1
    return zone

def geo_to_utm(lat, lon, zone):
    if (lat is None):
        raise ValueError("Invalid parameter: lat is null")
    if (lon is None):
        raise ValueError("Invalid parameter: lon is null")
    if (zone is None):
        zone = get_utm_zone(lon)
    geoProj = pyproj.Proj(proj="longlat", ellps="WGS84", datum="WGS84")
    utmProj = pyproj.Proj(proj="utm", zone=str(zone), ellps="WGS84", units="m")
    (x, y) = pyproj.transform(geoProj, utmProj, lon, lat)
    return (x, y, zone)

def utm_to_geo(x, y, zone):
    if (x is None):
        raise ValueError("Invalid parameter: x is null")
    if (y is None):
        raise ValueError("Invalid parameter: y is null")
    if (zone is None):
        raise ValueError("Invalid parameter: zone is null")
    utmProj   = pyproj.Proj(proj="utm", zone=str(zone), ellps="WGS84", units="m")
    geoProj = pyproj.Proj(proj="longlat", ellps="WGS84", datum="WGS84")
    (lon, lat) = pyproj.transform(utmProj, geoProj, x, y)
    return (lat, lon)

def max_lat_delta(lat, lon, dist_m):
    if (lat is None):
        raise ValueError("Invalid parameter: lat is null")
    if (lon is None):
        raise ValueError("Invalid parameter: lon is null")
    if (dist_m is None):
        raise ValueError("Invalid parameter: dist_m is null")
    (x, y, zn) = geo_to_utm(lat, lon, None)
    ya = y + dist_m
    yb = y - dist_m
    (lata, lon) = utm_to_geo(x, ya, zn)
    (latb, lon) = utm_to_geo(x, yb, zn)
    delta_a = abs(lat - lata)
    delta_b = abs(lat - latb)
    return max(delta_a, delta_b)

def max_lon_delta(lat, lon, dist_m):
    if (lat is None):
        raise ValueError("Invalid parameter: lat is null")
    if (lon is None):
        raise ValueError("Invalid parameter: lon is null")
    if (dist_m is None):
        raise ValueError("Invalid parameter: dist_m is null")
    (x, y, zn) = geo_to_utm(lat, lon, None)
    xa = x + dist_m
    xb = x - dist_m
    (lat, lona) = utm_to_geo(xa, y, zn)
    (lat, lonb) = utm_to_geo(xb, y, zn)
    delta_a = abs(lon - lona)
    delta_b = abs(lon - lonb)
    return max(delta_a, delta_b)

def max_deg_delta(lat, lon, dist_m):
    if (lat is None):
        raise ValueError("Invalid parameter: lat is null")
    if (lon is None):
        raise ValueError("Invalid parameter: lon is null")
    if (dist_m is None):
        raise ValueError("Invalid parameter: dist_m is null")
    mdlat = max_lat_delta(lat, lon, dist_m)
    mdlon = max_lon_delta(lat, lon, dist_m)
    return max(mdlat, mdlon)

# get postmile from geo coordinates
# align, rt, and/or rtsuf can be None if no restriction is desired
def getPmFromGeo(queryPt, rng, align, rt, rtsuf):
    if (queryPt is None):
        raise ValueError("Invalid parameter: queryPt is null")
    if (rng is None):
        raise ValueError("Invalid parameter: rng is null")
    resp = OpGeo2Pm.submitSoapQuery(queryPt, rng, align, rt, rtsuf)
    pm = OpGeo2Pm.parseSoapResult(resp)
    #if (pm is not None):
    #    pm.pmsuf = align
    return pm

# get geo coordinates from postmile
def getGeoFromPm(pm):
    if (pm is None):
        raise ValueError("Invalid parameter: pm is null")
    resp = OpPm2Geo.submitSoapQuery(pm)
    pt = OpPm2Geo.parseSoapResult(resp)
    return (pt)

