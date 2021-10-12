from typing import Optional, List, Dict
from pathlib import Path
import functools
import pickle
import io
import json

from tqdm import tqdm
import nltk
import lmdb
import lz4.frame as lz4
import numpy as np

from transformers import AutoTokenizer
import torch
from torch.utils import data


@functools.lru_cache()
def get_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@functools.lru_cache()
def get_word_to_id():
    cache_path = Path("word_to_id.json")
    if cache_path.exists():
        with open(cache_path, "r") as f:
            word_to_id = json.load(f)
    else:
        word_to_id = build_word_to_id()
        with open(cache_path, "w") as f:
            json.dump(word_to_id, f)

    return word_to_id


@functools.lru_cache()
def get_glove_weights():
    cache_path = Path("glove_weights.pth")
    if cache_path.exists():
        glove_weights = torch.load(cache_path, map_location="cpu")
    else:
        glove_weights = build_glove_weights(get_word_to_id())
        torch.save(glove_weights, cache_path)

    return glove_weights


class SparseTokensDataset(data.Dataset):
    def __init__(self, root_dir: str, num_sampled_frames: int = 32, split: str = "train"):
        self._root_dir = Path(root_dir)

        self._video_ids: Optional[List[str]] = None
        self._frame_ids: Dict[str, List[int]] = {}
        self._db: Optional[lmdb.Environment] = None

        self.num_sampled_frames = num_sampled_frames
        self._split = split
        if split == "train":
            self._prefix = "sparse_bert_tokens/packed_0"
        elif split == "val":
            self._prefix = "sparse_bert_tokens/val_packed_0"
        else:
            raise NotImplementedError

    @property
    def db(self) -> lmdb.Environment:
        if self._db is None:
            self._db = lmdb.open(
                str(self._root_dir),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                map_size=1099511627776 * 8,
            )

        return self._db

    @property
    def video_ids(self) -> List[str]:
        if self._video_ids is None:
            """
            prefix = "sparse_bert_tokens/packed_0/"
            with self.db.begin(write=False) as txn:
                video_ids = []
                for k in txn.cursor().iternext(values=False):
                    k = k.decode()
                    if not k.startswith(prefix):
                        continue
                    k = k[len(prefix):]
                    video_id = k[:6]
                    video_ids.append(video_id)
                    if "info" not in k[6:]:
                        fid = int(k.split("/")[1])
                        if video_id not in self._frame_ids:
                            self._frame_ids[video_id] = []
                        self._frame_ids[video_id].append(fid)
                self._video_ids = list(set(video_ids))
            torch.save((self._video_ids, self._frame_ids), "./cached_video_ids.pth")
            """
            self._video_ids, self._frame_ids = torch.load("./cached_video_ids.pth", map_location="cpu")
            # self._video_ids = [f"{i:06d}" for i in range(36199)]

        return self._video_ids

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        with self.db.begin(write=False) as txn:
            data = txn.get(f"{self._prefix}/{video_id}/info".encode())
            data = lz4.decompress(data)
            video_info = pickle.loads(data)

        frame_ids = self._frame_ids[video_id]
        # num_frames = video_info["video_frame_count"]
        # gt_temporal = torch.as_tensor(video_info["gt_temporal"], dtype=torch.float32)
        num_sampled_frames = self.num_sampled_frames

        stride = len(frame_ids) // self.num_sampled_frames
        sampled_frame_ids = [frame_ids[i * stride + stride // 2] for i in range(num_sampled_frames)]

        proposal_features = []
        with self.db.begin(write=False) as txn:
            for fid in sampled_frame_ids:
                data = txn.get(f"{self._prefix}/{video_id}/{fid:05d}/00".encode())
                buffer = io.BytesIO(data)
                proposal_features.append(torch.load(buffer, map_location="cpu"))

        proposal_features = torch.stack(proposal_features, dim=0)

        return video_id, video_info, sampled_frame_ids, proposal_features

    def __len__(self):
        return len(self.video_ids)


def collate(batch):
    vid_names = []
    batched_proposal_start_end = []
    batched_proposal_features = []
    batched_gt_temporal = []
    batched_query_tokens = []
    batched_query_len = []
    batched_num_sampled_frames = []
    batched_num_frames = []

    for _, video_info, _, _ in batch:
        sentence = video_info["sentence"]
        input_ids = sentence["input_ids"]
        query_len = (input_ids.index(102) if 102 in input_ids else len(input_ids)) - 1
        batched_query_len.append(query_len)

    sorted_batch = sorted(zip(batch, batched_query_len), key=lambda x: x[1], reverse=True)
    batched_query_len = []

    for (video_id, video_info, sampled_frame_ids, proposal_features), query_len in sorted_batch:
        vid_names.append(video_id)

        num_frames = video_info["video_frame_count"]
        gt_temporal = torch.as_tensor(video_info["gt_temporal"], dtype=torch.float32)
        proposal_start_end = torch.as_tensor([[fid, fid + 1] for fid in sampled_frame_ids], dtype=torch.float32)
        proposal_start_end = proposal_start_end / num_frames
        proposal_start_end = proposal_start_end.clamp(min=0, max=1.0)

        batched_proposal_start_end.append(proposal_start_end)
        batched_proposal_features.append(proposal_features)
        batched_gt_temporal.append(gt_temporal / num_frames)

        sentence = video_info["sentence"]
        input_ids = sentence["input_ids"]
        # query_len = (input_ids.index(102) if 102 in input_ids else len(input_ids)) - 1
        input_ids = input_ids[1:query_len + 1]
        tokenizer = get_tokenizer()
        word_to_id = get_word_to_id()
        query_words = nltk.word_tokenize(tokenizer.decode(input_ids).replace(".", ""))
        query_tokens = [word_to_id[word] for word in query_words]
        if len(query_tokens) < 20:
            query_tokens += [0 for _ in range(20 - len(query_tokens))]
        query_tokens = query_tokens[:20]
        assert len(query_tokens) == 20

        batched_query_tokens.append(query_tokens)
        batched_query_len.append(query_len)

        batched_num_sampled_frames.append(len(sampled_frame_ids))
        batched_num_frames.append(num_frames)

    batched_proposal_start_end = torch.stack(batched_proposal_start_end, dim=0)
    batched_proposal_features = torch.stack(batched_proposal_features, dim=0)
    batched_gt_temporal = torch.stack(batched_gt_temporal, dim=0)
    batched_query_tokens = torch.as_tensor(batched_query_tokens, dtype=torch.int64)
    batched_query_len = torch.as_tensor(batched_query_len, dtype=torch.int64)
    batched_num_sampled_frames = torch.as_tensor(batched_num_sampled_frames, dtype=torch.int64)
    batched_num_frames = torch.as_tensor(batched_num_frames, dtype=torch.int64)

    return vid_names, batched_proposal_start_end, batched_proposal_features, batched_gt_temporal, \
        batched_query_tokens, batched_query_len, batched_num_sampled_frames, batched_num_frames


def build_word_to_id():
    print("building word_to_id.")
    db = lmdb.open(
        "/home/shenenya/projs/datasets/pool",
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False
    )
    word_to_id = {}
    word_id = 1

    with db.begin(write=False) as txn:
        for video_id in tqdm(range(36199)):
            data = txn.get(f"sparse_bert_tokens/packed_0/{video_id:06d}/info".encode())
            data = lz4.decompress(data)
            video_info = pickle.loads(data)

            sentence = video_info["sentence"]
            input_ids = sentence["input_ids"]
            query_len = (input_ids.index(102) if 102 in input_ids else len(input_ids)) - 1
            input_ids = input_ids[1:query_len + 1]
            tokenizer = get_tokenizer()
            query_words = nltk.word_tokenize(tokenizer.decode(input_ids).replace(".", ""))
            for word in query_words:
                if word not in word_to_id:
                    word_to_id[word] = word_id
                    word_id += 1

    print("vocab_size:", len(word_to_id))
    return word_to_id


def build_glove_weights(word_to_id):
    full_glove = {}
    with open("datasets/glove.6B.300d.txt") as f:
        for line in list(f):
            values = line.strip("\n").split(" ")
            word = values[0]
            vector = torch.as_tensor([float(e) for e in values[1:]], dtype=torch.float32)
            full_glove[word] = vector

    weights = torch.zeros(len(word_to_id) + 1, 300, dtype=torch.float32)
    for word, idx in word_to_id.items():
        if word in full_glove:
            weights[idx] = full_glove[word]

    return weights


def main():
    # build_glove_weights
    get_glove_weights()

    ds = SparseTokensDataset(root_dir="/home/shenenya/projs/datasets/pool", num_sampled_frames=32)

    print(len(ds))

    vid_names, proposal_start_end, proposal_features, gt_temporal, \
        query_tokens, query_len, num_sampled_frames, num_frames = collate([ds[0]])

    print(proposal_start_end, proposal_start_end.shape)
    print(proposal_features.shape)
    print(gt_temporal, gt_temporal.shape)
    print(query_tokens, query_tokens.shape)
    print(query_len, query_len.shape)
    print(num_sampled_frames, num_sampled_frames.shape)
    print(num_frames, num_frames.shape)


if __name__ == "__main__":
    main()
