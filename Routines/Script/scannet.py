import JSONHelper
for r in JSONHelper.read('./full_annotations.json'):
    print(r["id_scan"])
