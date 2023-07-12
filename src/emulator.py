import time
import logging

class __TreeNode:
  def __init__(self, name=None, parent=None):
    self.name = str(id(self)) if name is None else name
    self.parent = parent
    self.left_siblings = []
    self.right_siblings = []
    self.children = []
    self._scan_siblings()

  @property
  def num_child(self):
    return len(self.children)

  def _append_child(self, child: "__TreeNode"):
    self.children.append(child)
    for child in self.children:
      child._scan_siblings()

  def _scan_siblings(self):
    if self.parent is None:
      return
    if self.parent.num_child == 0:
      return
    if not self in self.parent.children:
      return
    index_self = self.parent.children.index(self)
    self.left_siblings = self.parent.children[:index_self]
    self.right_siblings = self.parent.children[index_self+1:]

  def __repr__(self):
    return f'<{self.__class__.__name__}: {self.name}{"  num_child: " + str(self.num_child) if self.num_child > 0 else ""}  parent: {self.parent.name if not self.parent is None else None}>'


class TaskWrapper:
  verbose = True
  
  def __init__(self, name=None, host=None):
    self.name = str(id(self)) if name is None else name
    self.task_fn = None
    self.host = None
    self.host_logger : logging.Logger = None
    self.has_logger = False
    self.state = 'stopped'
    self.task_cfg = {
      'run_once': False,
    }
    self.set_host(host)

  def _bind_logger(self):
    if not hasattr(self.host, 'logger') or self.host.logger is None:
      return
    self.host_logger = self.host.logger
    self.has_logger = True

  def log_event(self, msg=''):
    if not self.has_logger:
      return
    self.host_logger.info(msg)

  def set_host(self, host: "UnitProcess"):
    self.host = host
    self._bind_logger()

  def set_task(self, task_fn):
    self.task_fn = task_fn

  def run_task(self, *args, **kwargs):
    if self.task_cfg.get('run_once') and self.state == 'finished':
      return
    self.state = 'running'
    self.task_fn(self, *args, **kwargs)
    self.state = 'finished'

  def __repr__(self):
    return f'<{self.__class__.__name__}: {self.name} state:  {self.state}>'


class UnitProcess(__TreeNode):
  def __init__(self, name=None, parent=None):
    super().__init__(name=name, parent=parent)
    self.task_queue = {}
    self.local_vars = {}
    self.logger = None

  def create_task(self, name=None):
    task = TaskWrapper(name=name)
    task.set_host(self)
    self.task_queue[task.name] = task
    return task

  def run_task(self, name, *args, **kwargs):
    if not name in self.task_queue:
      return
    kwargs.update({'host': self, 'local_vars': self.local_vars})
    self.task_queue[name].run_task(*args, **kwargs)

  def run_all_tasks(self, *args, **kwargs):
    for k in self.task_queue.keys():
      self.run_task(k, *kwargs, **kwargs)

  def spawn_child(self, name=None):
    child = self.__class__(name=name, parent=self)
    self._append_child(child=child)
    return child

  def spawn_self(self, name=None):
    o = self.__class__(name=name, parent=self.parent)
    if isinstance(self.parent, self.__class__):
      self.parent._append_child(child=o)
    else:
      raise ValueError("Could not create other instance of class on current tree node")
    return o

  def attach(self, other: "UnitProcess"):
    self._append_child(child=other)

  def get_child_by_name(self, name):
    for child in self.children:
      if child.name == name:
        return child
    return None

  def force_run_children_tasks(self, *args, **kwargs):
    for child in self.children:
      child.run_all_tasks(*args, **kwargs)

  def set_logger(self, logger):
    self.logger = logger

if __name__ == '__main__':
  root = UnitProcess()
  task_fn = lambda tw, host, local_vars: print(f'{tw=}, {host=}, {local_vars=}')
  for i in range(5):
    child = root.spawn_child()
    child.local_vars.update({'foo': i})

  for child in root.children:
    task = child.create_task()
    task.set_task(task_fn)

  root.force_run_children_tasks()