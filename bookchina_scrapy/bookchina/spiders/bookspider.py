# -*- coding: utf-8 -*-
import scrapy
from bookchina.items import BookchinaItem


class BookspiderSpider(scrapy.Spider):
    name = "bookspider"
    allowed_domains = ["http://www.bookschina.com"]
    start_urls = ['http://www.bookschina.com']
    hostname='http://www.bookschina.com'
    count=0

    def parse(self, response):
        first_category =response.css('div.categoryInner ul.category-list li.js_toggle div.category-info h3.category-name a::text').extract()
        for c1 in range(len(first_category)):
            second_category=response.css('div.categoryInner ul.category-list li.js_toggle:nth-child('+str(c1+1)+') div.category-info p.c-category-list a::text').extract()
            curr_first_category=response.css('div.categoryInner ul.category-list li.js_toggle:nth-child('+str(c1+1)+') div.category-info h3.category-name a::text')[0].extract()
            for c2 in range(len(second_category)):
                curr_second_category = response.css('div.categoryInner ul.category-list li.js_toggle:nth-child(' + str(
                    c1 + 1) + ') div.category-info p.c-category-list a:nth-child('+str(c2+1)+')::text')[0].extract()
                curr_link=self.hostname+response.css('div.categoryInner ul.category-list li.js_toggle:nth-child(' + str(
                    c1 + 1) + ') div.category-info p.c-category-list a:nth-child('+str(c2+1)+')::attr(href)')[0].extract()
                meta={}
                meta.update({'first_category':curr_first_category})
                meta.update({'second_category':curr_second_category})
                yield scrapy.Request(url=curr_link,callback=self.parse_info,dont_filter=True,meta=meta)

    def parse_info(self, response):
        item=BookchinaItem()
        info_tags=response.css('div#container.container div.listMain.clearfix div.listLeft div.bookList ul li')
        for tag in info_tags:
            item['name']=tag.css('div.infor h2.name a::text')[0].extract()
            item['pic_url']=tag.css('div.cover a img::attr(data-original)')[0].extract()
            item['author']=tag.css('div.infor div.otherInfor a.author::text')[0].extract()
            item['time']=tag.css('div.infor div.otherInfor span.pulishTiem::text')[0].extract().split()[0]
            item['publisher']=tag.css('div.infor div.otherInfor a.publisher::text')[0].extract()
            item['price']=tag.css('div.infor div.priceWrap span.sellPrice::text')[0].extract()
            item['first_cat']=response.meta['first_category']
            item['second_cat']= response.meta['second_category']
            print(item)
            yield item


        next_url = response.css('div.paging ul li.next a::attr(href)').extract()

        if next_url!=[]:
            #if int(next_url[0].split('_')[6]) <= 1:

            yield scrapy.Request(url=self.hostname+next_url[0],callback=self.parse_info,meta=response.meta,dont_filter=True)

