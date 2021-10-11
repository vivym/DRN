from typing import Optional, List
from pathlib import Path
import pickle

import lmdb
import lz4.frame as lz4

from torch.utils import data


class SparseTokensDataset(data.Dataset):
    def __init__(self, root_dir: str):
        self._root_dir = Path(root_dir)

        self._video_ids: Optional[List[str]] = None
        self._db: Optional[lmdb.Environment] = None

    @property
    def db(self) -> lmdb.Environment:
        if self._db is None:
            self._db = lmdb.open(
                str(self._root_dir),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

        return self._db

    @property
    def video_ids(self) -> List[str]:
        if self._video_ids is None:
            with self.db.begin(write=False) as txn:
                self._video_ids = set([k[:6].decode("ascii") for k in txn.cursor().iternext(values=False)])

        return self._video_ids

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        with self.db.begin(write=False) as txn:
            data = txn.get(f"{video_id}/info".encode("ascii"))
            data = lz4.decompress(data)
            video_info = pickle.loads(data)

        return video_info

    def __len__(self):
        return len(self.video_ids)


def main():
    ds = SparseTokensDataset(root_dir="/data3/yangming/features.lmdb")

    print(len(ds))

    print(ds[0])
    print(ds[10])


if __name__ == "__main__":
    main()
