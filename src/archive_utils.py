import zipfile
from pathlib import Path
from io import BytesIO

def unload_zip_from_bytes(_bytes, export_folder):
  export_folder = Path(export_folder)
  with zipfile.ZipFile(BytesIO(_bytes)) as zp:
    print('extracting zip file content:')
    for fd in zp.filelist:
      print(' ',
            f'size: {fd.file_size}',
            f'filename: {fd.filename}',
            sep='\t'
      )      
      zp.extractall(export_folder)