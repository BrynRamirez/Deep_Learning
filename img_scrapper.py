import os
import requests
from bs4 import *


# create folder
if not os.path.exists("penguins"):
    os.makedirs("penguins")

# download images
def dwnld(img, folder_name):
    count = 0
    if len(img) != 0:
        for i, image in enumerate(img):
            try:
                img_link = image["data-srcset"]
            except:
                try:
                    img_link = image["data-src"]
                except:
                    try:
                        img_link = image["data-fallback-src"]
                    except:
                        try:
                            img_link = image["src"]
                        except:
                            pass
            try:
                r = requests.get(img_link).content
                try:
                    r = str(r, 'utf-8')
                except UnicodeDecodeError:
                    with open(f"{folder_name}/images{i+173}.jpg", "wb+") as f:
                        f.write(r)
                        count += 1
            except:
                pass

            if count == len(img):
                print("Imgs downloaded")
            else:
                print(f"{count} Images downloaded out of {len(img)}")

url = 'https://www.google.com/search?q=single+penguin&tbm=isch&ved=2ahUKEwidjeyJnfb9AhW45ckDHTueCRQQ2-cCegQIABAA&oq=single+penguin&gs_lcp=CgNpbWcQAzIFCAAQgAQyBQgAEIAEMgUIABCABDIGCAAQCBAeMgYIABAIEB4yBggAEAgQHjIGCAAQCBAeMgcIABCABBAYOgQIIxAnOgcIABCKBRBDOggIABCABBCxAzoLCAAQgAQQsQMQgwFQjQZY5BRg-RVoAHAAeACAAWiIAaoJkgEEMTQuMZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=V3YeZN2FCLjLp84Pu7ymoAE&bih=979&biw=1920&rlz=1C1VDKB_enUS976US976'
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
images = soup.find_all('img')
dwnld(images, "penguins")