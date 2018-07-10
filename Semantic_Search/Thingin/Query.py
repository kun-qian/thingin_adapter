import requests
import json

Namespace = "TestForSniffer"
Hide_Default_Namespace = "false"
Content_Type = "application/json"
Accept = "application/json"
Query_Url = "http://ziggy-api-int.nprpaas.ddns.integ.dns-orange.fr/api/projections/find/"


def query_by_geo(latitude=116.409978, longtitude=39.968284, range=10000):
    my_headers = {'Namespace': Namespace, 'Hide-Default-Namespace': Hide_Default_Namespace,
                  'Accept': Accept, 'Content-Type': Content_Type}
    body = {"query": {
                "www_opengis_net_gml_pos": {
                    "$nearSphere": {
                        "$geometry": {"type": "Point", "coordinates": ""},
                        "$maxDistance": ""}}},
            "edge": "true", "view": {}
            }
    body["query"]["www_opengis_net_gml_pos"]["$nearSphere"]["$geometry"]["coordinates"] = [latitude, longtitude]
    body["query"]["www_opengis_net_gml_pos"]["$nearSphere"]["$maxDistance"] = range
    body_string = json.dumps(body)

    r = requests.post(Query_Url, headers=my_headers, data=body_string)

    if r.status_code == requests.codes.ok:
        print(r.content)

    return r.json()


query_by_geo()
