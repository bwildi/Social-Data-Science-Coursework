import wikipedia
import bs4
import mechanicalsoup
import pandas as pd
import re
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt
import time
import random

class GiveWellParser(mechanicalsoup.StatefulBrowser):
    """Mechanical soup stateful browser subclass
    specifically for scraping GiveWell's charities"""

    def __init__(self):
        super(GiveWellParser, self).__init__()
        self.open("https://www.givewell.org/")

    def ScrapeCharities(self, h="h3"):
        '''Function to scrape the names of charities when navigated
        onto a GiveWell charity page list. There is one parameter 
        - heading level - which is where the charity names are found'''
        heads = self.get_current_page().find_all(h)
        charities = []
        for head in heads:
            
            # Firstly try to find charities by looking for a url
            a = head.find("a")
            if a != None:
                
                # Sometimes the charity is stored as a title, 
                # sometimes as text
                try: entry = a["title"]
                except KeyError: entry = a.string
            
            # Sometimes there are no links so we 
            # just have to scrape headers
            else:

                # Remove the serachbar
                if head.string != "\n      Enter search terms here.\n    ":
                    entry = head.string
            
            # Remove parentheses - These are abbreviations or just
            # noting that GW only endorses the deworming aspect
            try:
                entry = re.sub(r"\([^)]*\)", "", entry)
                entry = re.sub(r"#\d", "", entry)
                entry = entry.strip()
                charities.append(entry)
            except (UnboundLocalError, TypeError): pass

        # Duplicate clean
        return list(set(charities))

def CharDataFrame(charities, years, months):
    i = 0
    char_dic = {"Name": [], "Year added": [], "Month added": []}
    for char_list in charities:
        a = []
        for charity in char_list:
            if charity in char_dic["Name"]:
                pass
            else:
                char_dic["Name"].append(charity)
                char_dic["Year added"].append(years[i])
                char_dic["Month added"].append(months[i])
        i += 1
    return pd.DataFrame(char_dic)

def GetWikiData(title):
    s = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    prop = "revisions"
    rvprop = "ids|timestamp|size|userid"
    rvlimit = "max"
    params = {"action": "query", "titles": title,
             "format": "json", "prop": prop, 
             "rvprop": rvprop, "rvlimit": rvlimit}
    revids, timestamps, size, userid = [], [], [], []
    r = s.get(url=url, params=params)
    data = r.json()["query"]["pages"]
    data = list(data.values())[0]["revisions"]
    for item in data:
        # remove bots
        if item["userid"] == 0: continue
        revids.append(item["revid"])
        size.append(item["size"])
        t = item["timestamp"]
        t = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
        timestamps.append(t)
        userid.append(item["userid"])
    
    df = pd.DataFrame({"Size": size, "Timestamp": timestamps, 
                        "user_id": userid}, index=revids)
    tf = []
    for i, uid in enumerate(df["user_id"]):
        if uid in df["user_id"].iloc[i + 1:].values: tf.append(False)
        else: tf.append(True)
    df["user_first_edit"] = tf 
    return df

def PlotPageHistory(s, t, ax1, year=None, month=None, title=""):
    cumul_revs = np.flip(np.arange(len(t)), axis=0)
    ax1.plot(t, s, 'b-')
    ax1.set_ylabel("Size", color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(t, cumul_revs, 'r-')
    ax2.set_ylabel("Revisions", color='r')
    ax2.tick_params('y', colors='r')
    if year != None:
        if month == np.nan: month = 1
        plt.axvline(x = datetime.datetime(year, month, 1))
    plt.title(title)

gwp = GiveWellParser()
gwp.follow_link("charities")
tc_2018 = gwp.ScrapeCharities()
print(tc_2018)

# Let's also get data on these guys from the previous years
gwp.follow_link("November-2016-version")
tc_2016 = gwp.ScrapeCharities()

# This contains some garbage, 
# but GiveWell is poorly formatted, 
# so not much we can do
tc_2016.remove("Charities Supporting Deworming")
tc_2016.remove("Other Charities Running\
 Cost-Effective Evidence-Backed Programs")

# June 2016
gwp.follow_link("June-2016-version")
tc_2016_jun = gwp.ScrapeCharities()
tc_2016_jun.remove("Other Standout Charities")

# 2015 - annoyingly they missed out
# the link for this one so we have to reset
gwp.follow_link("charities")
gwp.follow_link("November-2015-version")
tc_2015 = gwp.ScrapeCharities()
tc_2015.remove("Other Standout Charities")

# After this the formatting changes again, 
# but the list is very small and
# similar in previous years

# Note that this is true
tc_2016_jun == tc_2015

# So we can ignore the June data

top_charities = [tc_2015, tc_2016, tc_2018]
years = [2015, 2016, 2018]
months = [11] * 3
df = CharDataFrame(top_charities, years, months)

# Let's manually remove those extra Evidence Actions
df = df[df["Name"] != "Deworm the World Initiative"]
df = df[df["Name"] != "Deworm the World Initiative,\
 led by Evidence Action"]

# Need to do some manual year updates
df.at[9, "Year added"] = 2013
df.at[1, "Year added"] = 2012
df.at[2, "Year added"] = 2009
df.at[3, "Year added"] = 2011


# Add standout charities
# Also going back to 2015
gwp.follow_link("charities/other-charities")
s1 = gwp.ScrapeCharities("h4")
gwp.follow_link("october-2017-version")
s2 = gwp.ScrapeCharities("h4")
gwp.follow_link("february-2017-version")
s3 = gwp.ScrapeCharities("h4")
gwp.follow_link("2015-version")
s4 = gwp.ScrapeCharities("h4")

so_charities = [s4, s3, s2, s1]
years = [2015, 2017, 2017, 2018]
months = [np.nan, 2, 10, 6]
df2 = CharDataFrame(so_charities, years, months)

# Stick both dataframes together
df["Type"] = "Top"
df2["Type"] = "Standout"
gw_df = pd.concat((df, df2), ignore_index=True)

# Some of these need renaming
i = gw_df[gw_df["Name"] == "Iodine Global Network , \
                            formerly ICCIDD"].index[0]
gw_df["Name"].loc[i] = "Iodine Global Network"
i = gw_df[gw_df["Name"] == "The Global Alliance for\
 Improved Nutrition  - Universal Salt Iodization  program"].index[0]
gw_df["Name"].loc[i] = "The Global  Alliance for Improved Nutrition"

# HK is duplicated, since it was only added as a top charity
# in 2018 November, we will treat as standout
i = gw_df[gw_df["Name"] == "Helen Keller International"].index[0]
gw_df = gw_df.drop(i)
i = gw_df[gw_df["Name"] == "Helen Keller International \
 - Vitamin A Supplementation  program"].index[0]
gw_df["Name"].loc[i] = "Helen Keller International"

# Make name the index
gw_df = gw_df.set_index(gw_df["Name"])
gw_df = gw_df.drop(columns="Name")

# Let's look these up on wikipedia
names = []
for name in gw_df.index:
    try: names.append(wikipedia.search(name)[0])
    except IndexError: names.append(None)

gw_df["Wiki_Title"] = names

# Some of these look wrong or weren't found, 
# so did a manual check
# And removed entries who were not there
gw_df = gw_df.drop("END Fund")
gw_df = gw_df.drop("Food Fortification Initiative")
gw_df = gw_df.iloc[:-2]

# Now let's get data for each of these pages
revision_data = []
for title in gw_df["Wiki_Title"]:
    data = GetWikiData(title)
    revision_data.append(data)
    print("Success")
    time.sleep(1)

# Make final output plots
fig, axes = plt.subplots(6, 1)
for i in range(len(revision_data) // 2):
    s = revision_data[i]["Size"]
    t = revision_data[i]["Timestamp"]
    year = int(gw_df.iloc[i]["Year added"])
    try: month = int(gw_df.iloc[i]["Month added"])
    except: ValueError: month = ""
    title = gw_df.iloc[i]["Wiki_Title"]
    ax1 = axes[i]
    PlotPageHistory(s, t, ax1, year=year, 
    month=month, title=title)

plt.subplots_adjust(hspace=1)
plt.show()

fig, axes = plt.subplots(6, 1)
for i in range(len(revision_data) // 2, len(revision_data)):
    s = revision_data[i]["Size"]
    t = revision_data[i]["Timestamp"]
    year = int(gw_df.iloc[i]["Year added"])
    try: month = int(gw_df.iloc[i]["Month added"])
    except: ValueError: month = ""
    title = gw_df.iloc[i]["Wiki_Title"]
    ax1 = axes[i - len(revision_data) // 2]
    PlotPageHistory(s, t, ax1, year=year,
     month=month, title=title)

plt.subplots_adjust(hspace=1)
plt.show()

# We're also going to get a dozen random charities from wikipedia's list
# of international charities
browser = mechanicalsoup.StatefulBrowser()
browser.open("https://en.wikipedia.org/wiki/Category:International_charities")
random.seed(10)
print(len(browser.get_current_page().find_all('a')))

# There are 235 links on the page, 
# we're going to randomly select some of them
# and bin irrelevant other links 
# that exist on the page
random_charity_links = []
for i in range(8):
    k = browser.get_current_page().find_all('a')[random.randint(0, 234)]
    random_charity_links.append(k)

# Check if they ping out valid charity names, 
# otherwise ditch them
rnd_char_names = []
for link in random_charity_links:
    try: 
        rnd_char_names.append(link["title"])
    except KeyError: del link
print(rnd_char_names)
# Now we can use the same method to get their wikipedia data
# Now let's get data for each of these pages
revision_data2 = []
for title in rnd_char_names:
    data = GetWikiData(title)
    revision_data2.append(data)
    print("Success")
    time.sleep(1)

# Make output plots
fig, axes = plt.subplots(6, 1)
for i in range(len(revision_data2)):
    s = revision_data2[i]["Size"]
    t = revision_data2[i]["Timestamp"]
    title = rnd_char_names[i]
    ax1 = axes[i]
    PlotPageHistory(s, t, ax1, title=title)

plt.subplots_adjust(wspace=0.3, hspace=0.8)
plt.show()

# Could also do some comparisons on size and numbers of edits
# Look at if there's time
# s = requests.Session()
# url = "https://en.wikipedia.org/w/api.php"
# prop = "info"
# title = "GiveDirectly"
# params = {"action": "query", "titles": title, "format": "json", "prop": prop}
# r = s.get(url=url, params=params)
# data = r.json()["query"]["pages"]["33834107"]["length"]