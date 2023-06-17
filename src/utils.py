from datetime import datetime

def dict_to_flat_list(d: dict, ls=[]):
  for i in d.values():
    if isinstance(i, dict):
      dict_to_flat_list(i, ls)
    else:
      ls.append(i)
  return ls

def timestamp():
  return datetime.now().strftime("%Y%m%d-%H%M%S")

