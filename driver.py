from lib2to3.pgen2 import driver
from selenium import webdriver
import time
import sys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pickle
options = Options()
options.add_experimental_option("useAutomationExtension", False)
options.add_experimental_option("excludeSwitches",["enable-automation"])
driver = webdriver.Chrome(executable_path=r"C:\Users\Jack\Downloads\chromedriver_win32\chromedriver.exe", options=options)
cookies = pickle.load(open(r"C:\Users\Jack\Downloads\cookies.pkl", "rb"))
driver.get('https://www.geoguessr.com/game/qXZsQeXfr94qgELK')
for cookie in cookies:
    driver.add_cookie(cookie)
driver.get('https://www.geoguessr.com/game/qXZsQeXfr94qgELK')
driver.fullscreen_window()
def rotate_canvas():
    main = driver.find_element_by_tag_name('main')
    for _ in range(0,5):
        action = webdriver.common.action_chains.ActionChains(driver)
        action.move_to_element(main) \
            .click_and_hold(main) \
            .move_by_offset(118, 0) \
            .release(main) \
            .perform()
for css in ['#__next > div > div > main > div > div > div > div > div > div > div > div:nth-child(1) > div > div.gmnoprint > svg > path:nth-child(1)','#__next > div > div > main > div > div > div > div > div > div > div > div:nth-child(1) > div > div.gmnoprint > svg > path:nth-child(3)','#__next > div > div > main > div > div > div > div > div > div > div > div:nth-child(1) > div > div.gmnoprint > svg > path:nth-child(2)','#__next > div > div > main > div > div > div > div > div > div > div > div:nth-child(1) > div > div.gmnoprint > svg > path:nth-child(2)','#__next > div > div > main > div > div > div > div > div > div > div > div:nth-child(1) > div > div.gmnoprint > svg > path:nth-child(1)','#__next > div > div > main > div > div > div.game-layout__top-hud','#__next > div > div > main > div > div > aside','#__next > div > div > main > div > div > div.game-layout__status','#__next > div > div > main > div > div > div.game-layout__guess-map']:
    element=driver.find_element(By.CSS_SELECTOR,css)
    driver.execute_script("""
    var element = arguments[0];
    element.parentNode.removeChild(element);
    """, element)
time.sleep(2)
driver.find_element(By.CSS_SELECTOR,"#__next > div > div > main > div > div > div.game-layout__panorama > div > div > div > div > div:nth-child(1) > div > div:nth-child(10) > div > canvas").screenshot("C:\\Users\\Jack\\Downloads\\data\\test\\FO\\testimage.png")
time.sleep(100)
driver.close()