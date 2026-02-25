from pathlib import Path


class StorageManager:
    def __init__(self, name, orig_dir=None):
        self.orig_dir = orig_dir or Path.cwd()
        self.simulations_dir = self.orig_dir / "simulations"
        self.sim_dir = self.simulations_dir / name
        self.storage_dir = self.sim_dir / "storage"
        self.site_dir = self.sim_dir / "www"
        self.simulations_dir.mkdir(exist_ok=True)
        self.sim_dir.mkdir(exist_ok=True)
        self.storage_dir.mkdir(exist_ok=True)
        self.site_dir.mkdir(exist_ok=True)
        self.epoch_dir = None

    def update(self, epoch):
        self.epoch_dir = self.storage_dir / f"{epoch:06d}"
        self.epoch_dir.mkdir(exist_ok=True)

    def __getstate__(self):
        return {
            'orig_dir': self.orig_dir,
            'simulations_dir': self.simulations_dir,
            'sim_dir': self.sim_dir,
            'storage_dir': self.storage_dir,
            'site_dir': self.site_dir,
            'epoch_dir': self.epoch_dir,
        }

    @staticmethod
    def from_state(state):
        obj = object.__new__(StorageManager)
        obj.orig_dir = state['orig_dir']
        obj.simulations_dir = state['simulations_dir']
        obj.sim_dir = state['sim_dir']
        obj.storage_dir = state['storage_dir']
        obj.site_dir = state['site_dir']
        obj.epoch_dir = state['epoch_dir']
        return obj

    def __setstate__(self, state):
        self.orig_dir = state['orig_dir']
        self.simulations_dir = state['simulations_dir']
        self.sim_dir = state['sim_dir']
        self.storage_dir = state['storage_dir']
        self.site_dir = state['site_dir']
        self.epoch_dir = state['epoch_dir']
