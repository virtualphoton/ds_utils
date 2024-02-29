from getpass import getpass
from pathlib import Path

from dotenv import dotenv_values, set_key

def get_dotenv_key(key: str) -> str:
    """
    Tries to find `key` in .env file. If not found, adds (or creates)
    value input by user to the .env
    
    :param key: key to find in .env
    """
    file = Path(".env")
    if not file.exists():
        file.touch()
    values = dotenv_values(file)
    if key not in values:
        value = getpass(f"Key {key} wasn't found in .env input it (won't be displayed): ")
        if not value:
            raise RuntimeError("No key was provided, aborting")
        
        set_key(file, key, value)
        values[key] = value
            
    return values["NEPTUNE_API_KEY"]
