# -*- coding: utf-8 -*-
import requests
import json
import re
import os
import time

host = open('D:\MyConfiguration\szj46941\workspace\es.txt', encoding='utf-8').readline()

scroll_id_url = host + '/quality/_search?scroll=1m&search_type=scan'
scroll_query_url = host + '/_search/scroll'

query1 = {
    "size": 10000,
    "query": {
        "match_all": {}
    }
}

query2 = {
    "scroll": "1m",
    "scroll_id": ""
}


def split_speed_result(speed_result):
    speed_split_patten = speed_result.split(";")
    speed_split = [speed.split(" ") for speed in speed_split_patten]
    return speed_split


content_xml_reg = '^.*<.*>.*$'
sub_xml_reg = '<.*>'
sub_symbol_reg = "[\s+\.\!\/_,$%^*()+\"\'\:\-\=\;]+|[+——！，。？?、~@#￥%……&*（）\：\；\‘\’\“\”\、\-\=]+|[A-Za-z0-9]+|[☆机器人访客☆]"


def filter(line):
    if len(line) > 100: return ''
    if re.match(content_xml_reg, line):
        line = re.sub(sub_xml_reg, '', line)
    line = re.sub(sub_symbol_reg, "", line)  # 去掉中英文符号
    line = ''.join(re.findall(r'[\u4e00-\u9fa5]', line))
    return line


def process_raw_data(source):
    contents = source['contents']
    directions = source['directions']

    qas = []

    last_direction = 0

    i = 0
    for direction in directions:
        if last_direction == direction:
            qas[len(qas) - 1] = qas[len(qas) - 1] + ',' + filter(contents[i])
        else:
            qas.append(filter(contents[i]))
            last_direction = direction
        i += 1
    return qas


def fetch():
    return ''


if __name__ == '__main__':
    o = open('D:\MyConfiguration\szj46941\PycharmProjects\machine-learning\\bot\datasets\chinese\chatdata', mode='w+',
             encoding='utf-8')
    r1 = requests.post(scroll_id_url, json=query1)
    query2['scroll_id'] = json.loads(r1.text)['_scroll_id']
    count = 0
    total = 1
    while count < total:
        start = time.time()
        r2 = json.loads(requests.post(scroll_query_url, json=query2).text)
        query2['scroll_id'] = r2['_scroll_id']
        total = r2['hits']['total']

        hits = r2['hits']['hits']
        for hit in hits:
            source = hit['_source']
            metadata = process_raw_data(source)
            o.write(json.dumps(metadata))
            o.write('\r\n')
            o.flush()
            count += 1
        print('fetch: {}, total: {}, time: {}'.format(10000, count, time.time() - start))
