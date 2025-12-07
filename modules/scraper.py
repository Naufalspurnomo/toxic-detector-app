import pandas as pd
import time
from datetime import datetime, timedelta
import re
import warnings

# Google Play Scraper
try:
    from google_play_scraper import Sort, reviews
except ImportError:
    reviews = None

# Selenium
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    webdriver = None

class DateParser:
    @staticmethod
    def parse_twitter_date(date_str):
        """Parse text like '2h', 'Dec 12', '12 Dec 2023' into datetime."""
        now = datetime.now()
        date_str = date_str.strip()
        
        # Relative time
        if 'm' in date_str and len(date_str) < 5: # minutes
            minutes = int(re.search(r'\d+', date_str).group())
            return now - timedelta(minutes=minutes)
        if 'h' in date_str and len(date_str) < 5: # hours
            hours = int(re.search(r'\d+', date_str).group())
            return now - timedelta(hours=hours)
        
        # Absolute time (Twitter usually display "MMM DD" for current year, "MMM DD, YYYY" for past)
        # Handle "Dec 12" -> Dec 12, Current Year
        try:
            return datetime.strptime(f"{date_str}, {now.year}", "%b %d, %Y")
        except ValueError:
            pass
            
        try:
            # Handle "Dec 12, 2022"
            return datetime.strptime(date_str, "%b %d, %Y")
        except ValueError:
            pass
            
        return now # Fallback

class BaseScraper:
    def scrape(self, **kwargs):
        raise NotImplementedError

class GPlayScraper(BaseScraper):
    def scrape(self, app_id, count=100, lang='id', country='id'):
        if not reviews:
            raise ImportError("google-play-scraper not installed")
            
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=count
        )
        
        data = []
        for r in result:
            data.append({
                'source': 'Google Play',
                'username': r['userName'],
                'content': r['content'],
                'date': r['at'],
                'score': r['score']
            })
        
        return pd.DataFrame(data)

class SeleniumTwitterScraper(BaseScraper):
    def __init__(self, headless=False):
        if not webdriver:
            raise ImportError("Selenium not installed")
            
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=options
        )
        self.wait = WebDriverWait(self.driver, 10)

    def login(self, username, password):
        try:
            self.driver.get("https://twitter.com/i/flow/login")
            time.sleep(5)
            
            # Username
            user_input = self.wait.until(EC.presence_of_element_located((By.NAME, "text")))
            user_input.send_keys(username)
            user_input.send_keys(Keys.RETURN)
            time.sleep(3)
            
            # Check for unusual activity check (phone/email verification) - Skipping for simple flow
            # Password
            try:
                pass_input = self.wait.until(EC.presence_of_element_located((By.NAME, "password")))
                pass_input.send_keys(password)
                pass_input.send_keys(Keys.RETURN)
            except:
                # Sometimes asks for username again or verification
                pass    
            
            time.sleep(5)
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False

    def scrape(self, keyword, limit=100, start_date=None, end_date=None):
        data = []
        try:
            # Search
            url = f"https://twitter.com/search?q={keyword}&src=typed_query&f=live"
            self.driver.get(url)
            time.sleep(5)
            
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while len(data) < limit:
                # Find Tweets
                articles = self.driver.find_elements(By.TAG_NAME, "article")
                
                for article in articles:
                    try:
                        # Extract Text
                        text_element = article.find_element(By.CSS_SELECTOR, "div[data-testid='tweetText']")
                        text = text_element.text
                        
                        # Extract Date
                        time_element = article.find_element(By.TAG_NAME, "time")
                        date_str = time_element.get_attribute("datetime") 
                        # Twitter datetime attr is ISO format (e.g. 2023-12-07T14:00:00.000Z)
                        dt_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        
                        # Date Filtering
                        if start_date and dt_obj.date() < start_date:
                            # Too old, logic says we should stop? Twitter search results aren't strictly ordered sometimes
                            # But generally 'Latest' tab is ordered.
                            # For safety, let's just skip, or check if we encountered many old dates.
                            continue
                            
                        if end_date and dt_obj.date() > end_date:
                            continue
                            
                        # Avoid duplicates
                        if not any(d['content'] == text for d in data):
                            data.append({
                                'source': 'Twitter',
                                'username': 'Unknown', # Parsing username is nice but optional
                                'content': text,
                                'date': dt_obj
                            })
                            
                        if len(data) >= limit:
                            break
                            
                    except Exception as e:
                        continue
                
                # Scroll
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                
        finally:
            self.driver.quit()
            
        return pd.DataFrame(data)

# Factory
class ScraperFactory:
    @staticmethod
    def get_scraper(platform):
        if platform == "Google Play":
            return GPlayScraper()
        elif platform == "Twitter":
            return SeleniumTwitterScraper(headless=False) # Visible for login manual help if needed
        return None
