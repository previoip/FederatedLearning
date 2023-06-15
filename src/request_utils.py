import requests
from io import BytesIO
from pathlib import Path
from urllib.request import url2pathname
from urllib.parse import urlparse
from re import compile as re_compile
from os.path import join as path_join
from os.path import pathsep, relpath

__re_invalid_chars = re_compile(r'[^\w_. -]')

def url_to_fp(url):
  url_parsed = urlparse(url)
  netloc_path, file_path = url2pathname(url_parsed.netloc), url2pathname(url_parsed.path)
  netloc_path = __re_invalid_chars.sub('-', netloc_path)
  if url_parsed.path == '/':
    return netloc_path.strip('/').strip('\\')
  else:
    return path_join(netloc_path, file_path).strip('/').strip('\\')

def get_request(url, caching_enable=True, cache_folder='cache', params=None) -> bytes:

  cache_folder = Path(cache_folder).resolve()
  cache_file = cache_folder / url_to_fp(url) 

  if caching_enable and cache_file.exists() and cache_file.is_file():
    print(f'using cached file {relpath(cache_file)}')
    content = cache_file.read_bytes()
  else:
    print(f'downloading file to {relpath(cache_file)}')
    req = requests.get(url, params=params)
    req.raise_for_status()
    content = req.content
    req.close()

  if caching_enable:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(content)

  return content 

