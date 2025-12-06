from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.support.ui import Select

def scrape_player_data(team_url, year, output_file_name):
    driver = webdriver.Chrome()
    driver.get(team_url)

    for attempt in range(5):
        try:
            time.sleep(3)
            accept_btn = driver.find_element(By.XPATH, "/html/body/div[5]/div/div[1]/div/button")
            if accept_btn:
                accept_btn.click()
            break
        except Exception as e:
            print(e)
            time.sleep(3)
            driver.refresh()

    wait = WebDriverWait(driver, 6)
    history = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="sub-navigation"]/ul/li[5]/a')))
    history.click()

    select_element = driver.find_element(By.ID, 'stageId')
    select = Select(select_element)
    select.select_by_visible_text(year)

    detailed = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="team-squad-archive-stats-options"]/li[5]')))
    detailed.click()

    df = pd.DataFrame()

    stats = {
        "Shots": ["Zones", "Situations", "Accuracy"],
        "Goals": ["Zones", "Situations"],
        "Passes": ["Length", "Type"],
        "Key passes": ["Length", "Type"],
        "Dribbles": [],
        "Possession loss": [],
        "Aerial": [],
        "Assists": [],
        "Tackles": [],
        "Interception": [],
        "Fouls": [],
        "Offsides": [],
        "Clearances": [],
        "Blocks": [],
        "Saves": []
    }


    def get_table(df):
        time.sleep(1)
        wait = WebDriverWait(driver, 6)
        table = wait.until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="player-table-statistics-head"]/tr'))
            )
        table = driver.find_element(By.ID, 'statistics-table-detailed')
        html = table.get_attribute("outerHTML")
        curr = pd.read_html(html)[0]
        if df.empty == False:
            cols_to_add = [c for c in list(curr.columns) if c not in list(df.columns) and c != 'Player.1']
            df = df.merge(curr[['Player.1'] + cols_to_add], on='Player.1', how='left')

        else:
            df = curr
        return df


    for stat in stats:
        select_element = driver.find_element(By.ID, 'category')
        select = Select(select_element)
        select.select_by_visible_text(stat)

        if stats[stat] != []:
            for category in stats[stat]:
                select_element = driver.find_element(By.ID, 'subcategory')
                select = Select(select_element)
                select.select_by_visible_text(category)
                df = get_table(df)
        else:
            df = get_table(df)
    
    df.to_csv(output_file_name, index=False)
    driver.quit()

"""scrape_player_data(
    "https://www.whoscored.com/teams/30/show/england-tottenham", 
    "Premier League - 2023/2024", 
    "manunited2425")"""


"""
Instructions:
1. For the first argument, you need to go to whoscored.com then search
for your team and paste the link to that team, e.g. https://www.whoscored.com/teams/32/show/england-manchester-united

2. Then, you need to include the name of the season and league you want. 
You may need to go to a team's page and go to History -> Previous seasons to see examples of names.
E.g. Ligue 1 - 2024/2025, LaLiga 2023/2024

3. The third argument is the name of the file to output data

NOTE: this uses Selenium Chrome webdriver, so it will automatically cause
a new Chrome tab to open. You can just put it in the background and should extract 
the data.

However, the program is pretty buggy so you might need to check back and relaunch if it fails.

"""