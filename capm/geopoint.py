#
# Copyright (C) 2016 The Regents of the University of California
# All Rights Reserved
# Created by the Advanced Highway Maintenance and Construction
# Technology Research Center (AHMCT)
#

class GeoPoint(object):

    def __init__(self, lat, lon):
        if (lat is None):
            raise ValueError("Invalid parameter: lat is null")
        if (lon is None):
            raise ValueError("Invalid parameter: lon is null")
        if ((lat < -90.0) or (lat > 90.0)):
            raise ValueError("Invalid parameter: lat out of bounds")
        if ((lon < -180.0) or (lon > 180.0)):
            raise ValueError("Invalid parameter: lon out of bounds")
        self.lat    = float(lat)
        self.lon    = float(lon)

    def __str__(self):
        return "%f,%f" % (self.lat, self.lon)

