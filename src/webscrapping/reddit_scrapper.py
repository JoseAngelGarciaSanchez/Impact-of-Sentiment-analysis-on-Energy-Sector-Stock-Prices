from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from time import sleep
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import random


def create_browser_with_blocked_notifications():
    chrome_options = Options()
    prefs = {"profile.default_content_setting_values.notifications": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(options=chrome_options)
    return driver


class RedditBot:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.driver = create_browser_with_blocked_notifications()
        self.list_seconds = [1, 2, 3]  # Example values, adjust as needed
        self.verbose = True  # Print scroll progress information

    def accept_cookies(self):
        try:
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        '//*[@id="SHORTCUT_FOCUSABLE_DIV"]/div[3]/div/section/div/section[2]/section[1]/form/button',
                    )
                )
            ).click()
        except Exception as e:
            print(f"Cookie acceptance failed: {e}")

    def login(self):
        driver = self.driver
        driver.maximize_window()
        driver.get("https://www.reddit.com/login")
        time.sleep(2)

        username_elem = driver.find_element("xpath", '//*[@id="login-username"]')
        password_elem = driver.find_element("xpath", '//*[@id="login-password"]')

        username_elem.clear()
        password_elem.clear()

        username_elem.send_keys(self.username)
        password_elem.send_keys(self.password)
        password_elem.send_keys(Keys.RETURN)
        time.sleep(2)



    def search_and_scrape(self, search_query):
        self.driver.get(f"https://www.reddit.com/search/?q={search_query}&sort=new")
        sleep(2)
        
        last_position = self.driver.execute_script("return window.pageYOffset;")
        end_of_scroll_region = False
        
        with open(f"reddit_comments_{search_query}.csv", "w", newline="", encoding="utf-8") as file:
            fieldnames = ["Title", "Published Date", "Comment"]
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            
            while not end_of_scroll_region:
                posts = self.driver.find_elements(By.CSS_SELECTOR,'[data-testid="search-post"]')
                print(posts)
                for post in posts:
                    post_text = post.get_attribute("innerText")
                    post_lines = [
                        line.strip()
                        for line in post_text.split("\n")
                        if line.strip() and not any(unwanted in line for unwanted in ["MODO", "Mod", "•"])
                    ]                    
                    title = post_lines[0] if len(post_lines) > 0 else ""
                    published_date = post_lines[1] if len(post_lines) > 1 else ""
                
                    print("Got the data")
                    sleep(5)
                    print("Now trying to click")
                    to_comments = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, '//*[@id="main-content"]/div/reddit-feed/faceplate-tracker[1]'))
                    )
                    to_comments.click()
                    
                    print("Clicking worked")
                    sleep(50)
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '//div[@data-testid="comment"]'))
                    )
                    
                    comments = self.driver.find_elements(By.XPATH, '//div[@data-testid="comment"]')
                    for comment in comments:
                        comment_text = comment.text.split('\n')[0]  # Taking the first line of the comment as an example
                        writer.writerow({"Title": title, "Published Date": published_date, "Comment": comment_text})

                    self.driver.back()
                    sleep(random.choice(self.list_seconds))





    def _scroll_down_page(
        self, driver, last_position, scroll_attempt=0, max_attempts=3
    ):
        end_of_scroll_region = False
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(random.choice(self.list_seconds))
        curr_position = driver.execute_script("return window.pageYOffset;")
        if curr_position == last_position:
            if scroll_attempt < max_attempts:
                sleep(2)  # Wait a bit longer and try again
                last_position, end_of_scroll_region = self._scroll_down_page(
                    driver, last_position, scroll_attempt + 1, max_attempts
                )
            else:
                end_of_scroll_region = True
        else:
            last_position = curr_position
            if self.verbose:
                print("Scrolled down the page.")
        return last_position, end_of_scroll_region

    def close_browser(self):
        self.driver.close()


# Use the bot
username = "mosefdata"
password = "DataMosef2@"
reddit_bot = RedditBot(username, password)
sleep(2)
reddit_bot.login()
sleep(2)
reddit_bot.search_and_scrape("S%26P+500+energy")
