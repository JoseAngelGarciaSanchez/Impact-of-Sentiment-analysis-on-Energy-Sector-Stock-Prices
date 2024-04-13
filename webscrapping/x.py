import csv
from datetime import date, datetime
import os
import random
import urllib
from time import sleep

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
)
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import Chrome


class TwitterScrapper:
    """
    X/Twitter Scrapper for 2024
    """

    def __init__(
        self,
        research: str = "Elon musk",
        link: str = "https://twitter.com/i/flow/login",
        start_date: date = None,
        end_date: date = None,
        mail=None,
        username=None,
        password=None,
        verbose: bool = False,
    ) -> None:
        self.research = research
        self.link = link

        self.start_date = start_date
        self.end_date = end_date

        self.mail = mail
        self.username = username
        self.password = password

        self.verbose = verbose

    def _open_set_up_chrome(self):
        """
        Opens and sets up chrome in incognito mode (
        private navigation), maximazing the windows for avoiding changes
        in xpath which are relative to the size of the windows
        due to proportionality CSS methods.
        """
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--incognito")
        driver = Chrome(options=options)

        sleep(random.randint(1, 3))

        if self.verbose:
            print("Chrome opened")

        return driver

    def _open_twitter(self, driver, link):
        driver.get(link)
        sleep(random.randint(3, 5))

        if self.verbose:
            print("Linked opened")

        return driver

    def _init_twitter_session(self, driver):
        # Introduce mail
        try:
            connection_button = driver.find_element(
                "xpath",
                '//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input',
            )
            connection_button.click()
            connection_button.send_keys(self.mail + Keys.ENTER)
        except NoSuchElementException:
            if self.verbose:
                print("Mail textbox not found")
        sleep(random.randint(2, 4))

        # Introduce username
        try:
            username_textbox = driver.find_element(
                "xpath",
                '//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div[2]/label/div/div[2]/div/input',
            )
            username_textbox.send_keys(self.username + Keys.ENTER)
        except NoSuchElementException:
            if self.verbose:
                print("Username textbox not found")
        sleep(random.randint(2, 4))

        # Introduce password
        try:
            password_textbox = driver.find_element(
                "xpath",
                '//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input',
            )
            password_textbox.send_keys(self.password + Keys.ENTER)
        except NoSuchElementException:
            if self.verbose:
                print("Username textbox not found")

    def _twitter_query(self, driver, query):
        try:
            query_textbox = driver.find_element(
                "xpath",
                '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[2]/div/div[2]/div/div/div/div[1]/div/div/div/form/div[1]/div/div/div/label/div[2]/div/input',
            )
            query_textbox.send_keys(query + Keys.ENTER)
        except NoSuchElementException:
            if self.verbose:
                print("Query textbox not found")

    def _advanced_link(self, driver, query):
        """This enables us to search directly by the URL, avoiding many XPATH that could change, it allows you to specify some advanced parameters for the search"""
        company_name = query
        encoded_name = urllib.parse.quote(company_name)
        if self.end_date is not None and self.start_date is not None:
            advanced_link = f"https://twitter.com/search?f=liveq&={encoded_name}%20until%3A{self.end_date}%20since%3A{self.start_date}&src=typed_query"
        elif self.end_date is None and self.start_date is not None:
            advanced_link = f"https://twitter.com/search?f=live&q={encoded_name}%20since%3A{self.start_date}&src=typed_query"
        elif self.end_date is not None and self.start_date is None:
            advanced_link = f"https://twitter.com/search?f=live&q={encoded_name}%20until%3A{self.end_date}&src=typed_query"
        else:
            advanced_link = (
                f"https://twitter.com/search?q={encoded_name}&src=typed_query&f=live"
            )
        driver.get(advanced_link)
        if self.verbose:
            print("good for advanced link")
        return driver

    def _get_recents(self, driver):
        """get recents tweets from the 'lasts' page for recent activity"""
        try:
            recent = driver.find_element(
                "xpath",
                '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[1]/div[1]/div[2]/nav/div/div[2]/div/div[2]/a/div/div',
            )
            recent.click()
            if self.verbose:
                print("good for getting recents tweets")
        except NoSuchElementException:
            if self.verbose:
                print("Recents box not found")

    def __get_tweet_data(self, card):
        """Extract data from tweet card"""
        try:
            handle = card.find_element("xpath", './/span[contains(text(), "@")]').text
        except NoSuchElementException:
            return
        try:
            username = card.find_element("xpath", ".//span").text
        except NoSuchElementException:
            return
        try:
            postdate = card.find_element("xpath", ".//time").get_attribute("datetime")
        except NoSuchElementException:
            return
        try:
            comment = card.find_element("xpath", ".//div[2]/div[2]/div[1]").text
        except NoSuchElementException:
            return
        try:
            responding = card.find_element("xpath", ".//div[2]/div[2]/div[2]").text
        except NoSuchElementException:
            return
        try:
            text = comment + responding
        except NoSuchElementException:
            return
        try:
            reply_cnt = card.find_element("xpath", './/div[@data-testid="reply"]').text
        except NoSuchElementException:
            return
        try:
            retweet_cnt = card.find_element(
                "xpath", './/div[@data-testid="retweet"]'
            ).text
            # retweet_cnt = card.find_element('xpath','.//div[@data-testid="retweet"]').text
        except NoSuchElementException:
            return
        try:
            like_cnt = card.find_element("xpath", './/div[@data-testid="like"]').text
            # like_cnt = card.find_element('xpath','.//div[@data-testid="like"]').text
        except NoSuchElementException:
            return
        tweet = (username, handle, postdate, text, reply_cnt, retweet_cnt, like_cnt)

        return tweet

    def __generate_tweet_id(self, tweet):
        if self.verbose:
            print("good for generating tweets id")
        return "".join(tweet)

    def __collect_all_tweets_from_current_view(self, driver, lookback_limit=25):
        page_cards = driver.find_elements("xpath", '//article[@data-testid="tweet"]')
        if self.verbose:
            print("Good for collecting all tweets")

        if len(page_cards) <= lookback_limit:
            return page_cards
        else:
            return page_cards[-lookback_limit:]

    def __scroll_down_page(
        self, driver, last_position, scroll_attempt=0, max_attempts=5
    ):
        """The function will try to scroll down the page and will check the current
        and last positions as an indicator. If the current and last positions are the same after `max_attempts`
        the assumption is that the end of the scroll region has been reached and the `end_of_scroll_region`
        flag will be returned as `True`"""
        end_of_scroll_region = False

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        sleep(random.randint(3, 6))

        curr_position = driver.execute_script("return window.pageYOffset;")

        if curr_position == last_position:
            if scroll_attempt < max_attempts:
                if self.verbose:
                    print(f"max scroll attemps reached: {max_attempts}")
                end_of_scroll_region = True
            else:
                self.__scroll_down_page(
                    last_position, curr_position, scroll_attempt + 1
                )

        last_position = curr_position

        if self.verbose:
            print("good for scrolling down the page")

        return last_position, end_of_scroll_region

    def __save_tweet_data_to_csv(self, records, save_path: os.PathLike | str):
        """
        Saving the data collected into the
        specified save path with csv format
        """

        header = [
            "User",
            "Handle",
            "PostDate",
            "TweetText",
            "ReplyCount",
            "RetweetCount",
            "LikeCount",
        ]

        dire_path = os.path.dirname(save_path)
        file_path = f'{save_path}webscraped_{"_".join(self.research.split())}.csv'
        first = False
        if not os.path.exists(file_path):
            os.makedirs(dire_path, exist_ok=True)
            open(file_path, "a").close()
            first = True
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if first:
                writer.writerow(header)
            if records:
                writer.writerow(records)

    def _scroll_and_save(self, driver, save_path):

        last_position = None
        end_of_scroll_region = False
        unique_tweets = set()

        self.__save_tweet_data_to_csv(None, save_path)

        while not end_of_scroll_region:
            cards = self.__collect_all_tweets_from_current_view(driver)

            for card in cards:
                try:
                    tweet = self.__get_tweet_data(card)
                except StaleElementReferenceException:
                    continue
                if not tweet:
                    continue

                tweet_id = self.__generate_tweet_id(tweet)

                if tweet_id not in unique_tweets:
                    unique_tweets.add(tweet_id)

                    self.__save_tweet_data_to_csv(tweet, save_path)

            sleep(random.randint(3, 6))

            last_position, end_of_scroll_region = self.__scroll_down_page(
                driver, last_position
            )

        driver.quit()

    def launch_webscrapping(self, save_path: str | os.PathLike):
        if not save_path.endswith("/"):
            save_path += "/"

        driver = self._open_set_up_chrome()
        driver = self._open_twitter(driver=driver, link=self.link)

        self._init_twitter_session(driver=driver)
        sleep(random.randint(20, 40))

        self._advanced_link(driver, query=self.research)
        # self._twitter_query(driver, query=self.research)
        sleep(random.randint(5, 8))

        self._get_recents(driver)
        sleep(random.randint(6, 10))

        self._scroll_and_save(driver, save_path)

        sleep(120)

        return "finish"


if __name__ == "__main__":
    import os
    from time import sleep, time

    from dotenv import load_dotenv
    import pandas as pd

    load_dotenv()
    MAIL_1 = os.getenv('MAIL_1')
    USERNAME_1 = os.getenv('USERNAME_1')
    PASSWORD_1 = os.getenv('PASSWORD_1')

    MAIL_2 = os.getenv('MAIL_2')
    USERNAME_2 = os.getenv('USERNAME_2')
    PASSWORD_2 = os.getenv('PASSWORD_2')

    MAIL_3 = os.getenv('MAIL_3')
    USERNAME_3 = os.getenv('USERNAME_3')
    PASSWORD_3 = os.getenv('PASSWORD_3')

    # RESEARCH = "bp plc"
    RESEARCH = "stora enso"
    SAVE_PATH = "./../data/new_webscrapping/"
    FILE_PATH = f'{SAVE_PATH}webscraped_{"_".join(RESEARCH.split())}.csv'

    # df = pd.read_csv(FILE_PATH)
    # observations = df.shape[0]

    iteration = 1

    accounts = [
        (MAIL_1, USERNAME_1, PASSWORD_1),
        (MAIL_2, USERNAME_2, PASSWORD_2),
        (MAIL_3, USERNAME_3, PASSWORD_3),
    ]
    while True:
        start_time = time()
        for MAIL, USERNAME, PASSWORD in accounts:
            print(f"iteration: {iteration}")
            print(f"Current time: {datetime.now().time()}")

            if os.path.exists(FILE_PATH):
                df = pd.read_csv(FILE_PATH)
                minimale_date = pd.to_datetime(df["PostDate"]).min().date()
                observations = df.shape[0]

                print(f"Number of observations: {observations}")

                if pd.notna(minimale_date):
                    END_DATE = str(minimale_date)
                else:
                    END_DATE = None
            else:
                END_DATE = None

            print(f"END_DATE: {END_DATE}")

            ts = TwitterScrapper(
                research=RESEARCH,
                mail=MAIL,
                username=USERNAME,
                password=PASSWORD,
                # start_date=,
                end_date=END_DATE,
                verbose=False,
            )

            ts.launch_webscrapping(save_path=SAVE_PATH)
            iteration += 1

        passed_time = time() - start_time
        if passed_time < 1200:
            print(f"Need to sleep: {(1200-passed_time)/60} min")
            sleep(1200 - passed_time)
