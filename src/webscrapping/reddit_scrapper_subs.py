import csv
import random
import time
from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


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
        self.list_seconds = [1, 2, 3]
        self.verbose = True

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
        self.driver.get(f"https://www.reddit.com/r/{search_query}/new/")
        sleep(2)

        last_position = self.driver.execute_script("return window.pageYOffset;")
        end_of_scroll_region = False

        with open(
            f"subreddit_{search_query}.csv", "w", newline="", encoding="utf-8"
        ) as file:
            fieldnames = ["Title", "Author", "Published date", "Topic", "Link"]
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()

            while not end_of_scroll_region:
                post_html_elements = self.driver.find_elements(By.TAG_NAME, "article")
                unwanted_strings = ["MODO", "Mod", "•"]

                for post_html_element in post_html_elements:
                    post_text = post_html_element.get_attribute("innerText")
                    post_lines = [
                        line.strip()
                        for line in post_text.split("\n")
                        if line.strip()
                        and not any(unwanted in line for unwanted in unwanted_strings)
                    ]

                    post_dict = {
                        "Title": post_lines[1] if len(post_lines) > 0 else "",
                        "Author": post_lines[0] if len(post_lines) > 1 else "",
                        "Published date": post_lines[2] if len(post_lines) > 2 else "",
                        "Topic": "",
                        "Link": "",
                    }

                    if len(post_lines) > 3:
                        post_dict["Topic"] = " ".join(post_lines[3:-1])
                        # Add link if the last line starts with "http"
                        if post_lines[-1].startswith("http"):
                            post_dict["Link"] = post_lines[-1]

                    writer.writerow(post_dict)

                last_position, end_of_scroll_region = self._scroll_down_page(
                    self.driver, last_position
                )

    def _scroll_down_page(
        self, driver, last_position, scroll_attempt=0, max_attempts=3
    ):
        end_of_scroll_region = False
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(random.choice(self.list_seconds))
        curr_position = driver.execute_script("return window.pageYOffset;")
        if curr_position == last_position:
            if scroll_attempt < max_attempts:
                sleep(2)
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
reddit_bot.search_and_scrape("BP_PLC")
reddit_bot.close_browser()
