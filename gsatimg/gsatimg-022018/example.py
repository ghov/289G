#!/usr/bin/env python3

import sys
import io
import http.client

import gsatimg


def main(argv):

    lat = 38.528824
    lon = -121.756951
    zoom = 18

    pngdata = gsatimg.getpng(lat, lon, zoom)
    if (pngdata is None):
        print("error")
    else:
        gsatimg.savepng(pngdata, "out.png")


if __name__ == "__main__":
    global cmdname
    cmdname = sys.argv[0]
    main(sys.argv[1:])

