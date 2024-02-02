import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import _save_openai_configuration, _save_azure_openai_configuration
import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("./config/default_key.json")


# Define your check_api_key function here
def check_api_key():
    os.environ['show_api_setting'] = 'False'
    ## load json file
    with open(DEFAULT_CONFIG_PATH) as f:
        data = json.load(f)

    if "openai_key" in data:
        if _save_openai_configuration(data["openai_key"], False):
            print("default openai key changed\n\n")
            os.environ['show_api_setting'] = 'True'
            os.environ['default_openai_key'] = data["openai_key"]
        else:
            print("default openai key changed FAILED\n\n")
            os.environ.pop('default_openai_key', None)
    if "azure_key" in data:
        if _save_azure_openai_configuration(data["azure_key"],
                                            data["azure_base"], False):
            print("default azure key changed\n\n")
            os.environ['show_api_setting'] = 'True'
            os.environ['default_azure_key'] = data["azure_key"]
            os.environ['default_azure_base'] = data["azure_base"]
        else:
            print("default azure key changed FAILED!\n\n")
            os.environ.pop('default_azure_key', None)
            os.environ.pop('default_azure_base', None)


class FileChangeHandler(FileSystemEventHandler):

    def on_modified(self, event):
        if event.src_path == DEFAULT_CONFIG_PATH.__str__():
            check_api_key()


def start_observer():
    time.sleep(5)
    check_api_key()

    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=Path("./config"), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
