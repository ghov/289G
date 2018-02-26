#
# Copyright (C) 2016 The Regents of the University of California
# All Rights Reserved
# Created by the Advanced Highway Maintenance and Construction
# Technology Research Center (AHMCT)
#

import http.client
import urllib
import xml.dom.minidom

from . import const
from . import pmutil
from .pm import Pm

from .op import Op
from .op import ContractError
from .op import SoapParseError


class OpGeo2Pm(Op):

    @staticmethod
    def submitSoapQuery(queryPt, rng_ft, align, rt, rtsfx):
        if (queryPt is None):
            raise ContractError("Invalid parameter: queryPt is null")
        if (rng_ft is None):
            raise ContractError("Invalid parameter: rng_ft is null")
        if (rng_ft < 0):
            raise ContractError("Invalid parameter: rng_ft is invalid")
        srid = "4269"	# NAD83
        tol_deg = pmutil.max_deg_delta(queryPt.lat, queryPt.lon, float(rng_ft) / const.FEET_PER_METER)
        body = (
            '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"\n'
            '               xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
            '               xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n'
            '  <soap:Body>\n'
            '    <getPostmileForPointParameters xmlns="urn:webservice.postmile.lrs.gis.dot.ca.gov">\n'
            '      <inputPoint>\n'
            '        <m>0</m>\n'
            '        <spatialReferenceID>' + srid + '</spatialReferenceID>\n'
            '        <x>' + Op.xmlEsc(queryPt.lon) + '</x>\n'
            '        <y>' + Op.xmlEsc(queryPt.lat) + '</y>\n'
            '        <z>0</z>\n'
            '      </inputPoint>\n'
        )
        if (tol_deg is not None):
            body += (
            '      <options>\n'
            '        <toleranceDegrees>' + Op.xmlEsc(tol_deg) + '</toleranceDegrees>\n'
            '      </options>\n'
            )
        if (rt is not None):
            body += '      <routeNumber>' + Op.xmlEsc(rt) + '</routeNumber>\n'
        else:
            body += '      <routeNumber xsi:nil="true"/>\n'
        if (rtsfx is not None):
            body += '      <routeSuffixCode>' + Op.xmlEsc(rtsfx) + '</routeSuffixCode>\n'
        else:
            body += '      <routeSuffixCode xsi:nil="true"/>\n'
        if (align is not None):
            body += '      <routeAlignment>' + Op.xmlEsc(align) + '</routeAlignment>\n'
        else:
            body += '      <routeAlignment xsi:nil="true"/>\n'
        body += (
            '    </getPostmileForPointParameters>\n'
            '  </soap:Body>\n'
            '</soap:Envelope>\n'
        )
        headers = {
            "Host":const.PMSERVICE_HOST,
            "Content-Type":"text/xml",
            "Content-Length":len(body),
            "SOAPAction": "GetPostmileForPoint"
            }
        conn = http.client.HTTPConnection(const.PMSERVICE_HOST)
        conn.request("POST", const.PMSERVICE_PATH, body, headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return data


    @staticmethod
    def parseSoapResult(soapdoc):
        if (soapdoc is None):
            raise ValueError("Invalid parameter: soapdoc is null")

        doc = xml.dom.minidom.parseString(soapdoc)

        nrp_z    = None
        nrp_y    = None
        nrp_m    = None
        nrp_x    = None
        nrp_srid = None
        align    = None
        cty      = None
        pmpfx    = None
        pm       = None
        rt       = None
        rtsfx    = None
        try:
            envElem  = Op.getFirstElem(doc,      "soapenv:Envelope")
            bodyElem = Op.getFirstElem(envElem,  "soapenv:Body")
            rtnElem  = Op.getFirstElem(bodyElem, "getPostmileForPointReturn")
            dfrElem  = Op.getFirstElem(rtnElem,  "distanceFromRoute")
            nrpElem  = Op.getFirstElem(rtnElem,  "nearestRoutePoint")
            opmElem  = Op.getFirstElem(rtnElem,  "outputPostmile")
            dfr      = float(Op.getFirstChildData(dfrElem))
            if (not (Op.isNil(nrpElem))):
                nrp_z    = float(Op.getChildData(nrpElem, "z"))
                nrp_y    = float(Op.getChildData(nrpElem, "y"))
                nrp_m    = float(Op.getChildData(nrpElem, "m"))
                nrp_x    = float(Op.getChildData(nrpElem, "x"))
                nrp_srid = int(  Op.getChildData(nrpElem, "spatialReferenceID"))
            if (not (Op.isNil(opmElem))):
                align    = Op.getChildData(opmElem, "alignmentCode")
                cty      = Op.getChildData(opmElem, "countyCode")
                pmpfx    = Op.getChildData(opmElem, "postmilePrefixCode")
                pm       = Op.getChildData(opmElem, "postmileValue")
                rt       = Op.getChildData(opmElem, "routeNumber")
                rtsfx    = Op.getChildData(opmElem, "routeSuffixCode")
        except SoapParseError as e:
            #print "SoapParseError: " + e.msg
            #print soapdoc
            return None
        finally:
            doc.unlink()
        if (cty is None):
            #print "error"
            return None
        if (rt is None):
            #print "error"
            return None
        if (pm is None):
            #print "error"
            return None
        crp = (
            cty +
            "-" +
            rt  +
            (rtsfx if (rtsfx is not None) else ("")) +
            "-" +
            (pmpfx if (pmpfx is not None) else ("")) +
            pm +
            (align if (align is not None) else (""))
        )
        crp = str(crp)	# unicode
        pm = Pm(crp)
        return pm

