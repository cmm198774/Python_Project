# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class BookchinaItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    author=scrapy.Field()
    name=scrapy.Field()
    pic_url=scrapy.Field()
    first_cat=scrapy.Field()
    second_cat=scrapy.Field()
    price=scrapy.Field()
    publisher=scrapy.Field()
    time=scrapy.Field()

