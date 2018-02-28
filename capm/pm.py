#
# Copyright (C) 2016 The Regents of the University of California
# All Rights Reserved
# Created by the Advanced Highway Maintenance and Construction
# Technology Research Center (AHMCT)
#

from . import const
import re


# postmile
class Pm(object):

    # CRP string format: [CTY]-[RT][RTSFX]-[PMPFX][PM][PMSFX]
    def __init__(self, crp):
        if (type(crp) is not str):
            raise ValueError("crp must be a string")
        self.cty    = None
        self.rt     = None
        self.rtsfx  = None
        self.pmpfx  = None
        self.pmval  = None
        self.pmsfx  = None
        crp_comp = crp.split("-")
        if (len(crp_comp) != 3):
            raise ValueError("crp does not contain 3 hyphen-delimited fields")
        cty_full = crp_comp[0]
        rt_full  = crp_comp[1]
        pm_full  = crp_comp[2]
        if (cty_full not in const.CA_CTYS):
            raise ValueError("Error parsing county field")
        self.cty = cty_full
        if (re.match("^\d\d?\d?(?:[SU])?$", rt_full) is None):
            raise ValueError("Error parsing route field")
        rt_comp = re.split("([SU])", rt_full)
        self.rt = int(rt_comp[0])
        if (len(rt_comp) > 1):
            self.rtsfx = rt_comp[1]
        if (re.match("^(?:[CDGHLMNRST])?\d+(?:\.\d+)?(?:[RLX])?$", pm_full) is None):
            raise ValueError("Error parsing postmile field")
        pm_a = re.split("(\d+(?:\.\d+)?(?:[RLX])?)", pm_full, 1)
        if (len(pm_a) < 2):
            raise ValueError("Error parsing postmile field")
        self.pmpfx = (None if (pm_a[0] == "") else pm_a[0])
        pm_b = re.split("([RLX])", pm_a[1], 1)
        if (len(pm_b) < 1):
            raise ValueError("Error parsing postmile field")
        self.pmval = float(pm_b[0])
        if (len(pm_b) > 1):
            self.pmsfx = pm_b[1]

    # construct postmile from parameters
    @staticmethod
    def mkpm(cty, rt, rtsfx, pmpfx, pm, pmsfx):
        if ((cty is not None) and (type(cty) is not str)):
            raise ValueError("Invalid parameter: cty")
        if ((rt is not None) and (type(rt) is not int)):
            raise ValueError("Invalid parameter: rt")
        if ((rtsfx is not None) and (type(rtsfx) is not str)):
            raise ValueError("Invalid parameter: rtsfx")
        if ((pmpfx is not None) and (type(pmpfx) is not str)):
            raise ValueError("Invalid parameter: pmpfx")
        if ((pm is not None) and (type(pm) is not float)):
            raise ValueError("Invalid parameter: pm")
        if ((pmsfx is not None) and (type(pmsfx) is not str)):
            raise ValueError("Invalid parameter: pmsfx")
        return Pm(
            cty,
            rt,
            rtsfx,
            pmpfx,
            pm,
            pmsfx
        )

    def __str__(self):
        rs = "%s-%s%s-%s%.3f%s" % (
            self.cty,
            self.rt,
            self.rtsfx if self.rtsfx is not None else "",
            self.pmpfx if self.pmpfx is not None else "",
            float(self.pmval),
            self.pmsfx if self.pmsfx is not None else ""
        )
        return rs

