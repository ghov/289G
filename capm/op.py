#
# Copyright (C) 2016 The Regents of the University of California
# All Rights Reserved
# Created by the Advanced Highway Maintenance and Construction
# Technology Research Center (AHMCT)
#

class ContractError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)


class SoapParseError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)

# abstract base class
class Op(object):

    @staticmethod
    def xmlEsc(s):
        esc = str(s)
        # order is consequential
        esc = esc.replace("&",  "&amp;")
        esc = esc.replace(">",  "&gt;")
        esc = esc.replace("<",  "&lt;")
        esc = esc.replace("'",  "&apos;")
        esc = esc.replace("\"", "&quot;")
        return esc;

    @staticmethod
    def getFirstElem(node, name):
        if (node is None):
            raise ContractError("node is None")
        elems = node.getElementsByTagName(name)
        if (len(elems) < 1):
            raise SoapParseError("element \"" + name + "\" not found")
        return elems[0]

    @staticmethod
    def getFirstChildData(node):
        if (node is None):
            raise ContractError("node is None")
        if (len(node.childNodes) < 1):
            return None
        return node.childNodes[0].data

    # returns as string, or None
    @staticmethod
    def getChildData(parentElem, childName):
        if (parentElem is None):
            raise ContractError("parentElem is None")
        elem = Op.getFirstElem(parentElem, childName)
        if (elem.getAttribute("xsi:nil") == "true"):
            return None
        return Op.getFirstChildData(elem)

    @staticmethod
    def isNil(elem):
        if (elem is None):
            raise ContractError("elem is None")
        if (elem.getAttribute("xsi:nil") == "true"):
            return True
        return False

