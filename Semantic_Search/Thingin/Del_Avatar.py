import logging

import requests

from Thingin.Query import query_by_geo

Namespace = "TestForSniffer"
Accept = "application/json"
Del_Base_Url = "http://ziggy-api-int.nprpaas.ddns.integ.dns-orange.fr/api/projections/"

def del_a_avatar(uuid):
    my_headers = {'Namespace': Namespace, 'Accept': Accept}

    url = Del_Base_Url + uuid
    r = requests.delete(url, headers=my_headers)

    if r.status_code != 204:
        logging.debug("error in delete avatar : " + uuid)
        return False
    else:
        logging.debug("succeed in delete avatar : " + uuid)
        return True


def del_avatars():
    query_results = query_by_geo()
    for a_avartar in query_results["items"]:
        del_a_avatar(a_avartar["_uuid"])


del_avatars()