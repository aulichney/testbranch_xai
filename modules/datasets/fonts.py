from pathlib import Path
import requests, time
import zipfile

def get_fonts(fonts=["Open Sans", "Roboto", "Acme", "Lato", "Teko", "Ubuntu"], folder="./data/fonts"):
    """
    download open fonts

    Args:
        fonts (list, optional): list of fonts to download. Defaults to ["Open Sans", "Roboto", "Acme", "Lato", "Teko", "Ubuntu"].
        folder (str, optional): path to download the fonts to. Defaults to "./data/fonts".

    Raises:
        RuntimeError: raise error if the fonts fail to download
    """
    if isinstance(fonts, str): fonts = [fonts]
    # create base dir if doesn't exits
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    for font in fonts:
        # verify if exist:
        if len(list(Path(folder).glob("**/{font}*.ttf"))) == 0:
            # request
            url = requests.utils.requote_uri(f"https://fonts.google.com/download?family={font.replace(' ', '%20')}")
            for tries in range(3):
                r = requests.get(url)
                if r.status_code==200:
                    break
                time.sleep(1)
            if r.status_code!=200: raise RuntimeError(f"Failed to download:{r.text}")
            # download
            temp_zip = (Path(folder)/'fonts.zip').as_posix()
            with open(temp_zip, 'wb') as f:
                f.write(r.content)
            #extract
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall((Path(folder)/font).as_posix())
            Path(temp_zip).unlink()

if __name__ == '__main__':
    get_fonts()