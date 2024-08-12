# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

import logging
import os
import subprocess
import urllib
from pathlib import Path
import json
import requests
import torch
from src.labelme.labelme.utils import netio


def is_url(url, check=True):
    # Check if string is URL and check if URL exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def url_getsize(url='https://ultralytics.com/images/bus.jpg'):
    # Return downloadable file size in bytes
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get('content-length', -1))


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    # Could anyone explain what are url1 and url2?
    # Are such comments pure nonsense?
    try:  # url1
        LOGGER.info(f'Downloading {url} to {file}...')
        content = netio.secure_download(url)
        with open(file, 'wb') as f:
            f.write(content)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.info(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        # TLSv1.2 is now required.
        os.system(
            f"curl --tlsv1.2 -# -L '{url2 or url}' -o '{file}' --retry 3 -C -")
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info('')


def attempt_download(file, repo='ultralytics/yolov5', release='v7.0'):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v7.0', etc.
    from utils.general import LOGGER

    def github_assets(repository, version='latest'):
        # Return GitHub repo tag (i.e. 'v7.0') and assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        tag = 'v7.0'
        assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']
        return tag, assets

    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1E5)
            return file

        # GitHub assets
        assets = [f'yolov5{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')]  # default
        try:
            tag, assets = github_assets(repo, release)
            print(tag)
            print(assets)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            url3 = 'https://drive.google.com/drive/folders/1EFQTEUeXWSFww0luse2jB9M1QNZQGwNl'  # backup gdrive mirror
            with open("../../../../../../version.json", "r") as download_urls:
                try:
                    url = json.load(download_urls)["model_urls"]["yolov5"] + tag + "/" + name
                    safe_download(
                        file,
                        url,
                        min_bytes=1E5,
                        error_msg=f'{file} missing,\
                         try downloading from https://github.com/{repo}/releases/{tag} or {url3}')
                except Exception as e:
                    print(e)

    return str(file)
