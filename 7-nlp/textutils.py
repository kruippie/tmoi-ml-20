import requests
import re
import os

def get_file(datafile, url):
    files = os.listdir()
    if datafile not in files:
        req = requests.get(url)
        url_content = req.content
        csv_file = open(datafile, 'wb')
        csv_file.write(url_content)
        csv_file.close()


def flatten_json(nested_json, exclude=['']):
    """Flatten json object with nested keys into a single level.
        Args:
            nested_json: A nested json object.
            exclude: Keys to exclude from output.
        Returns:
            The flattened json object if successful, None otherwise.
    """
    out = {}

    def flatten(x, name='', exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude: flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out


def remove_url(text):
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', text)

def bin_time(time):
    if time < "2017-12-01":
        return 0
    elif time < "2018-01-01":
        return 1
    elif time < "2018-08-10":
        return 2
    elif time < "2019-08-01":
        return 3
    else:
        return 4