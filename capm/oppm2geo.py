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
from .geopoint import GeoPoint
from .op import Op
from .op import ContractError
from .op import SoapParseError


class OpPm2Geo(Op):

    @staticmethod
    def submitSoapQuery(pm):
        if (pm is None):
            raise ContractError("Invalid parameter: pm is null")
        if (pm.pmval is None):
            raise ContractError("Invalid parameter: pm.pmval is null")

        body = (
            '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"\n'
            '               xmlns:q0="urn:webservice.postmile.lrs.gis.dot.ca.gov"\n'
            '               xmlns:xsd="http://www.w3.org/2001/XMLSchema"\n'
            '               xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
            '  <soap:Body>\n'
            '    <q0:getCoordinatesForPostmileParameters>\n'
            '      <q0:options>\n'
            '        <q0:alignmentType>0</q0:alignmentType>\n'
            '        <q0:offsetDistance>0.0</q0:offsetDistance>\n'
            '      </q0:options>\n'
            '      <q0:postmileEvent>\n'
        )
        if ((pm.pmsfx == "L") or (pm.pmsfx == "R")):
            body += '        <q0:alignmentCode>' + Op.xmlEsc(pm.pmsfx) + '</q0:alignmentCode>\n'
        else:
            body += '        <q0:alignmentCode xsi:nil="true" />\n'
        if (pm.cty is not None):
            body += '        <q0:countyCode>' + Op.xmlEsc(pm.cty) + '</q0:countyCode>\n'
        else:
            body += '        <q0:countyCode xsi:nil="true" />\n'
        if (pm.pmpfx is not None):
            body += '        <q0:postmilePrefixCode>' + Op.xmlEsc(pm.pmpfx) + '</q0:postmilePrefixCode>\n'
        else:
            body += '        <q0:postmilePrefixCode xsi:nil="true" />\n'
        body += '        <q0:postmileValue>' + Op.xmlEsc(pm.pmval) + '</q0:postmileValue>\n'
        if (pm.rt is not None):
            body += '        <q0:routeNumber>' + Op.xmlEsc(pm.rt) + '</q0:routeNumber>\n'
        else:
            body += '        <q0:routeNumber xsi:nil="true" />\n'
        if (pm.rtsfx is not None):
            body += '        <q0:routeSuffixCode>' + Op.xmlEsc(pm.rtsfx) + '</q0:routeSuffixCode>\n'
        else:
            body += '        <q0:routeSuffixCode xsi:nil="true" />\n'
        body += (
            '      </q0:postmileEvent>\n'
            '      <q0:postmileSegmentEvent xsi:nil="true"/>\n'
            '    </q0:getCoordinatesForPostmileParameters>\n'
            '  </soap:Body>\n'
            '</soap:Envelope>\n'
        )

        headers = {
            "Host":const.PMSERVICE_HOST,
            "Content-Type":"text/xml",
            "Content-Length":len(body),
            "SOAPAction": "getCoordinatesForPostmile"
            }
        conn = httplib.HTTPConnection(const.PMSERVICE_HOST)
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
        lat = None
        lon = None
        try:
            envElem  = Op.getFirstElem(doc,      "soapenv:Envelope")
            bodyElem = Op.getFirstElem(envElem,  "soapenv:Body")
            rtnElem  = Op.getFirstElem(bodyElem, "getCoordinatesForPostmileReturn")
            pgElem   = Op.getFirstElem(rtnElem,  "pointGeometry")
            lat = float(Op.getChildData(pgElem, "y"))
            lon = float(Op.getChildData(pgElem, "x"))
        except SoapParseError as e:
            #print "SoapParseError: " + e.msg
            #print soapdoc
            return None
        finally:
            doc.unlink()
        if ((lat is None) or (lon is None)):
            return None
        return GeoPoint(lat, lon)

