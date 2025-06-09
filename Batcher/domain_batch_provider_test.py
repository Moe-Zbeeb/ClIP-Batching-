import os, sys, tempfile, shutil, math
import h5py, numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from domain_batch_provider import DomainBatchDataset

def create_domain_h5(dir_path, domain_name, num_samples):
    file_path = os.path.join(dir_path, f"{domain_name}.h5")
    with h5py.File(file_path, 'w') as f:
        grp = f.create_group(domain_name)
        dt_bytes = h5py.vlen_dtype(np.dtype('uint8'))
        ds_img = grp.create_dataset('images', shape=(num_samples,), dtype=dt_bytes)
        for i in range(num_samples):
            ds_img[i] = np.full((5,), i, dtype='uint8')
        dt_str = h5py.string_dtype('utf-8')
        ds_cap = grp.create_dataset('captions', shape=(num_samples,), dtype=dt_str)
        ds_cap[:] = [f"{domain_name}_{i}" for i in range(num_samples)]
    return file_path


def main():
    base = tempfile.mkdtemp()
    d = os.path.join(base, 'h5dir')
    os.makedirs(d)
    create_domain_h5(d, 'alpha', 5)
    create_domain_h5(d, 'beta', 3)
    ds = DomainBatchDataset(source=d, batch_size=2)
    batches = list(ds)
    expected = math.ceil(5/2) + math.ceil(3/2)
    assert len(batches) == expected
    print(f"Passed: {len(batches)} batches as expected")
    shutil.rmtree(base)

if __name__ == '__main__':
    main()
