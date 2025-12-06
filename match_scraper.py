from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
import time
import re
import os


# Get links to all matches in a season
def get_match_links(driver, url):
    driver.get(url)
    time.sleep(2)

    rows = driver.find_elements(By.CSS_SELECTOR, f"#matchlogs_for > tbody > tr")
    links, names = [], []
    for row in rows[:10]:
        cells = row.find_elements(By.TAG_NAME, "th")
        for cell in cells:
            a_tags = cell.find_elements(By.TAG_NAME, "a")
            if a_tags:
                links.append(a_tags[0].get_attribute("href"))
                date = a_tags[0]
        opponent = row.find_element(By.CSS_SELECTOR, f'td[data-stat="opponent"]')
        name = "match_" + date.text.replace("-", "") + "_" + opponent.text.replace(" ","")
        names.append(name)
    return(links, names)

def get_match_data(driver, link, new_folder_path, names, index):
    driver.get(link)
    time.sleep(2)
    table = driver.find_element(By.XPATH, f'//table[caption[text()="Tottenham Player Stats Table"]]')
    html = table.get_attribute("outerHTML")
    df = pd.read_html(html)[0]


    def flatten(col):
        a, b = col
        a = "" if "Unnamed" in str(a) else str(a).strip()
        b = "" if "Unnamed" in str(b) else str(b).strip()
        name = "_".join([x for x in [a, b] if x]).replace(" ", "_")
        return name

    df.columns = [flatten(c) for c in df.columns]

    # 3. Drop totals row and clean up
    df = df[df["Player"].astype(str).str.contains("Players") == False]
    df["Min"] = pd.to_numeric(df["Min"], errors="coerce")
    df = df[df["Min"] >= 30]

    name = names[index]
    full_filename = os.path.join(new_folder_path, f"{name}.csv")
    df.to_csv(full_filename, index=False)
    print("Saved matches.csv")

def scrape_season_fixtures(season_url, dummy, folder_name):
    base_dir = os.getcwd()
    new_folder_path = os.path.join(base_dir, folder_name)
    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Directory '{new_folder_path}' created or already exists.")
    except OSError as e:
        print(f"Error creating directory: {e}")
    
    driver = webdriver.Chrome()
    links, names = get_match_links(driver, season_url)
    for index, link in enumerate(links[0:10]):
        get_match_data(driver, link, new_folder_path, names, index)
    driver.quit

def get_match_results(link, output_file):
    driver = webdriver.Chrome()
    driver.get(link)
    table = driver.find_element(By.ID, "matchlogs_for")
    html = table.get_attribute("outerHTML")
    df = pd.read_html(html)[0]
    df = df[["Date", "Venue", "Result", "GF", "GA", "Opponent", "xG", "xGA", "Poss", "Formation", "Opp Formation"]]
    df = df[:10]
    df["Opponent"] = df["Opponent"].str.replace(" ", "", regex=False)
    df.to_csv(output_file, index=False)

def get_league_table(link, output_file):
    driver = webdriver.Chrome()
    driver.get(link)
    table = driver.find_element(By.ID, "results2024-202591_overall")
    html = table.get_attribute("outerHTML")
    df = pd.read_html(html)[0]
    df = df[["Rk", "Squad", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts", "Pts/MP", "xG", "xGA", "xGD", "xGD/90"]]
    df["Squad"] = df["Squad"].str.replace(" ", "", regex=False)
    df.to_csv(output_file, index=False)

# URL to all matches for a team in a season
my_url = "https://fbref.com/en/squads/361ca564/2022-2023/c9/Tottenham-Hotspur-Stats-Premier-League"
output_folder_name = "prem_22_23_matches"

#scrape_season_fixtures(my_url, None, output_folder_name)


