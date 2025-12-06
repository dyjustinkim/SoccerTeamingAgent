from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
import time
import re

# Get links to all matches in a season
def get_match_links(driver, url):
    driver.get(url)
    time.sleep(2)

    elem = driver.find_element(By.XPATH, '//div[@class="table_container tabbed current is_setup"]')
    elem_id = elem.get_attribute("id")[4:]
    rows = driver.find_elements(By.CSS_SELECTOR, f"#{elem_id} > tbody > tr")
    links = []
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(cells) >= 6:
            a_tags = cells[5].find_elements(By.TAG_NAME, "a")
            if a_tags:
                links.append(a_tags[0].get_attribute("href"))

    return(links)

def get_match_data(driver, link):
    driver.get(link)
    time.sleep(2)
    data = {}
    for num, team in enumerate(["team1", "team2"]):
        # Get team name
        elem = driver.find_element(By.XPATH, f'//*[@id="content"]/div[2]/div[{str(num+1)}]')
        name = elem.find_element(By.TAG_NAME, "a")
        data[team + " name"] = name.text
        # Get number of goals
        score = elem.find_element(By.CLASS_NAME, "score")
        data[team + " goals"] = score.text
        # Get expected goals (xG)
        xg = elem.find_element(By.CLASS_NAME, "score_xg")
        data[team + " xG"] = xg.text
        # Get formation
        letter = "a" if num == 0 else "b"
        elem2 = driver.find_element(By.ID, "field_wrap")
        lineup = elem2.find_element(By.ID, letter)
        formation = lineup.find_element(By.TAG_NAME, "th")
        match = re.search(r'\((.*?)\)', formation.text)
        if match:
            data[team + " formation"] = match.group(1)
        # Get possession
        poss = driver.find_element(By.XPATH, 
                                   f'//*[@id="team_stats"]/table/tbody/tr[3]/td[{str(num + 1)}]/div/div[1]/strong')
        data[team + " poss"] = poss.text
    return data

def scrape_season_fixtures(season_url, output_file_name):
    driver = webdriver.Chrome()
    links = get_match_links(driver, season_url)
    all_matches = []
    # Extract the first half of matches in a season for performance
    for link in links[0:190]:
        match = get_match_data(driver, link)
        all_matches.append(match)
    df = pd.DataFrame(all_matches)
    print(df)
    df.to_csv(f"evaluation/{output_file_name}.csv", index=False)
    driver.quit

# URL to a single season
my_url = "https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures"
scrape_season_fixtures(my_url, "Prem_22-23")

