import os
import glob
import math
import logging
import h5py

class DomainBatchDataset:
    def __init__(self, source, batch_size, domain_order_file=None):
        self.source = source
        self.batch_size = batch_size
        self.domain_order_file = domain_order_file
        if os.path.isdir(self.source):
            pattern = os.path.join(self.source, "*.h5")
            files = sorted(glob.glob(pattern))
            self._domain_files = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
        elif os.path.isfile(self.source) and self.source.endswith('.h5'):
            with h5py.File(self.source, 'r') as f:
                groups = list(f.keys())
            self._domain_files = {dom: self.source for dom in groups}
        else:
            raise ValueError(f"Source must be a .h5 file or directory: {self.source}")
        domains = list(self._domain_files.keys())
        if self.domain_order_file:
            with open(self.domain_order_file) as f:
                order = [line.strip() for line in f if line.strip()]
            self.domain_order = [d for d in order if d in domains]
            remaining = sorted(set(domains) - set(self.domain_order))
            self.domain_order += remaining
        else:
            self.domain_order = sorted(domains)
        logging.info(f"Initialized domains: {self.domain_order}")

    def __iter__(self):
        handles = {}
        lengths = {}
        for dom, path in self._domain_files.items():
            f = h5py.File(path, 'r')
            grp = f[dom] if dom in f else f
            handles[dom] = grp
            lengths[dom] = grp['images'].shape[0]
        max_batches = {dom: math.ceil(lengths[dom] / self.batch_size) for dom in self.domain_order}
        served = {dom: 0 for dom in self.domain_order}
        indices = {dom: 0 for dom in self.domain_order}
        active = list(self.domain_order)
        while active:
            for dom in list(active):
                if served[dom] >= max_batches[dom]:
                    active.remove(dom)
                    continue
                grp = handles[dom]
                start = indices[dom]
                end = start + self.batch_size
                total = lengths[dom]
                if end <= total:
                    imgs = grp['images'][start:end]
                    caps = grp['captions'][start:end]
                else:
                    imgs = list(grp['images'][start:total])
                    caps = list(grp['captions'][start:total])
                    pad_count = end - total
                    if total > 0:
                        imgs += [grp['images'][total-1]] * pad_count
                        caps += [grp['captions'][total-1]] * pad_count
                    else:
                        logging.warning(f"Domain '{dom}' has no samples; skipping batch.")
                        served[dom] = max_batches[dom]
                        continue
                indices[dom] = end if end < total else end - total
                served[dom] += 1
                imgs_bytes = [bytes(x) for x in imgs]
                caps_str = [c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else str(c) for c in caps]
                yield imgs_bytes, caps_str, dom
        for grp in handles.values():
            grp.file.close()
