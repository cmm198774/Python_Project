# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


class BookchinaPipeline(object):
    def process_item(self, item, spider):
        pic_url=item['pic_url']
        import requests
        import os
        import csv
        pic=requests.get(pic_url)
        path = os.getcwd() + '\\'
        if  not os.path.exists(path+'IMG'):
            os.makedirs(path+'IMG')
        with open(path+'IMG\\'+item['name']+'.'+pic_url[-3:],'wb') as f:
            f.write(pic.content)
        path = os.getcwd() + '\\'
        if not os.path.exists(path+'CSV'):
            os.makedirs(path + 'CSV')

        csvfile=open(path+'CSV\\scrapy_data.txt','a+',encoding='utf8')
        csvWriter=csv.writer(csvfile)
        #item['time']=item['time'].split('\\')[0]
        csvWriter.writerow([item['name'],item['author'],item['publisher'],item['price'],item['first_cat'],item['second_cat']])
        csvfile.close()

