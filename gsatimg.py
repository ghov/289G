#!/usr/bin/env python3

import io
import http.client

import const


def getpng(lat, lon, zoom):
    urlpath = (
        const.URLBASE +
        "?maptype=" + const.MAPTYPE +
        "&size=" + const.RESX + "x" + const.RESY +
        "&center=" + str(lat) + "," + str(lon) +
        "&zoom=" + str(zoom) +
        "&key=" + const.APIKEY
    )
    conn = http.client.HTTPSConnection(const.HOST)
    conn.request("GET", urlpath)
    resp = conn.getresponse()
    if (resp.status != 200):
        print(urlpath)
        print(resp.status, resp.reason)
        return None
    data = resp.read()
    return data

def savepng(pngdata, filename):
    with io.open(filename, 'wb') as f:
        f.write(pngdata)

