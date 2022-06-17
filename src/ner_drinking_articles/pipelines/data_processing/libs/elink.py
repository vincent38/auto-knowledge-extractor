import requests


def find_wikidata_id(item):
    try:
        url = "https://www.wikidata.org/w/api.php?" \
            f"action=wbsearchentities&search={item}" \
              "&language=en&format=json"
        data = requests.get(url).json()
        # Return the first id (Could upgrade this in the future)
        return data['search'][0]['id']
    except Exception:
        return 'id-less'


def find_duck_def(item):
    try:
        item = item.replace(' ', '%20')
        url = f'https://api.duckduckgo.com/?q={item}&format=json&no_html=1'
        data = requests.get(url).json()
        # print(data["RelatedTopics"])
        if data["Abstract"] != '':
            return data["Abstract"]
        else:
            txt = ""
            for i in range(0, len(data["Results"])):
                print(item, i)
                txt += "Hypothesis "+str(i)+": "
                txt += data["Results"][i]["Text"]
                txt += '- '
            return txt
    except Exception:
        return 'no-data'
