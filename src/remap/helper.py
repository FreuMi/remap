import sys
from rdflib import Graph, Namespace, URIRef

def getPath(graph: Graph) -> str:
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/path":
            return str(o)   

def getSubject(graph: Graph) -> tuple[str,str]:
    subjectMap = ""
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/subjectMap":
            subjectMap = str(o)
            break
    for s,p,o in graph:
        if str(s) == subjectMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
            
    print("Error in getSubject")
    sys.exit(1)
            
def getPredicate(graph: Graph) -> tuple[str,str]:
    predicateMap = ""
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/predicateMap":
            predicateMap = str(o)
            break
    
    for s,p,o in graph:
        if str(s) == predicateMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
    
    print("Error in getPredicate")
    sys.exit(1)
            
def getObject(graph: Graph) -> tuple[str,str]:
    objectMap = ""
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/objectMap":
            objectMap = str(o)
            break
    for s,p,o in graph:
        if str(s) == objectMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
    print("Error in getObject")
    sys.exit(1)

def getGraph(graph: Graph) -> tuple[str,str]:
    graphMap = ""
    for s,p,o in graph:
        if str(p) == "http://w3id.org/rml/graphMap":
            graphMap = str(o)
            break
    for s,p,o in graph:
        if str(s) == graphMap:
            if str(p) == "http://w3id.org/rml/constant":
                return str(o), "constant"
            elif str(p) == "http://w3id.org/rml/reference":
                return str(o), "reference"
            elif str(p) == "http://w3id.org/rml/template":
                return str(o), "template"
    return "", ""