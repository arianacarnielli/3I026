# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:34:52 2019

@author: Guilherme Mazanti
"""

import numpy as np
import requests as rq
import re
import time

tmdbid = np.load("tmdbid.npz")
tmdbid = tmdbid["tmdbid"]
URLBase = "https://www.themoviedb.org/movie/"
budgetRE = re.compile("<p><strong><bdi>Budget</bdi></strong>\s*\$([^<]*)</p>")
revenueRE = re.compile("<p><strong><bdi>Revenue</bdi></strong>\s*\$([^<]*)</p>")
titleRE = re.compile("<title>(.*) &#8212; The Movie Database \(TMDb\)</title>")

with open("budget.csv", "w", encoding="utf-8") as file:
  file.write("tmdbId;title;budget;revenue\n")
  
  for filmid in tmdbid:
    print(filmid)
    file.write(str(filmid)+";")
    
    page = rq.get(URLBase + str(filmid))
    assert page.status_code == 200 and page.ok
    
    title = titleRE.findall(page.text)
    budget = budgetRE.findall(page.text)
    revenue = revenueRE.findall(page.text)
    
    if len(title)!=1:
      print("WARNING: Title not found.")
    else:
      title = title[0]
      print(title)
      file.write(title.replace(";",","))
      
    file.write(";")
    
    if len(budget)!=1:
      print("WARNING: Budget not found.")
    else:
      budget = budget[0]
      print(budget)
      file.write(budget)
      
    file.write(";")
      
    if len(revenue)!=1:
      print("WARNING: Revenue not found.")
    else:
      revenue = revenue[0]
      print(revenue)
      file.write(revenue)
    
    print("")
    file.write("\n")
    
    file.flush()
    time.sleep(0.33)