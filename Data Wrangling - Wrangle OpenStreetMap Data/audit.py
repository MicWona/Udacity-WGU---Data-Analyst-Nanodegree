#!/usr/bin/env python
# coding: utf-8

# In[17]:


import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint


osm_file = open("pittsburgh_sample.osm", "r")

#Auditing Street Names
street_type_re = re.compile(r'\b\S+\.?$',re.IGNORECASE)
street_types = defaultdict(set)
     
expected_street = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons", "Circle", "Expressway", "Extension", "Highway", "Pike", 
            "Terrace", "Way"]

mapping = { "St": "Street",
            "Ave": "Avenue",
            "Rd": "Road"}

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected_street:
            street_types[street_type].add(street_name)

def print_sorted_dict(d):
    keys = d.keys()
    keys = sorted(keys, key=lambda s: s.lower())
    for k in keys:
        v = d[k]
        print "%s: %d" % (k, v)
                           
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def audit_streets(osm_file):
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types
    
# Clean street names   
def update_street_name(name):
    m = street_type_re.search(name)
    
    if m:
        street_type = m.group()
        if street_type not in expected_street: 
            if street_type in mapping:
                name = name.replace(street_type, mapping[street_type])
            
    return name


def test_street_name():
    street_types = audit_streets(osm_file)
    pprint.pprint(dict(street_types))

    for street_types, ways in street_types.iteritems():
        for name in ways:
            better_name = update_street_name(name)
            print name, "=>", better_name

            
#Used for testing function audit_postcode()
#if __name__ == '__main__':
 #   test_street_name()  
    
    
#Audit postcodes
postcode_type_re = re.compile(r'\b\S+\.?$',re.IGNORECASE)
postcode_types = defaultdict(set)

#Postcodes for Pitsburg found on https://worldpostalcode.com/united-states/pennsylvania/pittsburgh
expected_postcode = ['15201', '15202', '15203', '15204', '15205', '15206', '15207', '15208', '15209', '15210', '15211', 
                     '15212', '15213', '15214', '15215', '15216', '15217', '15218', '15219', '15220', '15221', '15222', 
                     '15223', '15224', '15225', '15226', '15227', '15228', '15229', '15230', '15231', '15232', '15233', 
                     '15234', '15235', '15236', '15237', '15238', '15239', '15240', '15241', '15242', '15243', '15244', 
                     '15250', '15251', '15252', '15253', '15254', '15255', '15257', '15258', '15259', '15260', '15261', 
                     '15262', '15264', '15265', '15267', '15268', '15270', '15272', '15274', '15275', '15276', '15277', 
                     '15278', '15279', '15281', '15282', '15283', '15286', '15289', '15290', '15295']

def audit_postcode_type(postcode_types, postcode):
    m = postcode_type_re.search(postcode)
    if m:
        postcode = m.group()
        if postcode not in expected_postcode:
            postcode_types[postcode].add(postcode)

def print_sorted_dict(d):
    keys = d.keys()
    keys = sorted(keys, key=lambda s: s.lower())
    for k in keys:
        v = d[k]
        print "%s: %d" % (k, v)     
            
def is_postcode(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit_postcode(osm_file):
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_postcode(tag):
                    audit_postcode_type(postcode_types, tag.attrib['v'])
    osm_file.close()
    return postcode_types

# Clean postcodes    
def update_postcodes(postcode):
    m = postcode_type_re.search(postcode)
    
    if m:
        postcode = m.group()
        if postcode not in expected_postcode:
            if len(postcode) > 5:
                postcode = postcode[0:5]
            else:
                postcode = postcode
    return postcode

def test_postcode():
    postcode_types = audit_postcode(osm_file)
    pprint.pprint(dict(postcode_types))

    for postcode_types, ways in postcode_types.iteritems():
        for postcode in ways:
            better_name = update_postcodes(postcode)
            print postcode, "=>", better_name

#Used for testing function audit_postcode()
#if __name__ == '__main__':
 #   test_postcode()


# In[ ]:




