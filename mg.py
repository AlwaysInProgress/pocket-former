import os
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import json
import cv2
from pytube import YouTube
from torch.utils.data import Dataset
import torch
import numpy as np
from utils import *
from typing_extensions import get_args

mg_path = 'data/mg/'

annotations_path = 'annotations'

if not os.path.exists(annotations_path):
    os.makedirs(annotations_path)

def mg_dir_path(id: int):
    path = mg_path + str(id) + '/'
    path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def mg_annotations_path(web_id: int):
    path = annotations_path + '/' + str(web_id) + '.json'
    path = os.path.join(os.path.dirname(__file__), path)
    return path

ALL_LABELS = Literal['moving', 'not_moving', 'inspection', 'scramble', 'unlabeled']

@dataclass
class MG:
    id: int # folder index
    web_id: int # index from website, we dont want to use this as the index
    url: str # video url
    moves: List[str] # list of notation moves
    # each entry is a frame number of when the cube starts moving or stops moving
    # even = start moving, odd = stop moving
    action_frames: List[Tuple[int, ALL_LABELS]] = field(default_factory=list)
    is_test: bool = False
    # Some videos look bad and we don't want them
    ignored: bool = False

    def add_label(self, frame_num: int, label: ALL_LABELS):
        self.action_frames.append((frame_num, label))

    def remove_last_action(self):
        self.action_frames.pop()

    @staticmethod
    def get_from_fs(id: int):
        path = mg_dir_path(id) + 'data.json'
        if not os.path.exists(path):
            print('Mg not found')
            return None

        with open(path, 'r') as f:
            data = f.read()
            print('Data:', data)
            mg = MG(**json.loads(data))

        annotations_path = mg_annotations_path(mg.web_id)
        if not os.path.exists(annotations_path):
            return mg
        with open(annotations_path, 'r') as f:
            data = f.read()
            annotations = json.loads(data)
            mg.action_frames = annotations["action_frames"]
            mg.is_test = annotations["is_test"]
            mg.ignored = annotations["ignored"]

        return mg

    def save_to_fs(self):
        print('Saving to fs: ', self.id)
        with open(mg_dir_path(self.id) + 'data.json', 'w') as f:
            s = json.dumps(self.__dict__)
            f.write(s)
        annotations_path = mg_annotations_path(self.web_id)
        with open(annotations_path, 'w') as f:
            # Make dict only with data we (the inganeers) manually annotated
            data = {
                "id": self.web_id,
                "action_frames": self.action_frames,
                "is_test": self.is_test,
                "ignored": self.ignored
            }
            s = json.dumps(data)
            f.write(s)
    
    def print(self):
        print("Id:", self.id)
        print("Url:", self.url)
        print("Moves:", self.moves)
        print("Action Frames:", self.action_frames)
        print("Is test:", self.is_test)

    def download_video(self, skip_if_exists: bool = True):
        if skip_if_exists and os.path.exists(mg_dir_path(self.id) + 'video.mp4'):
            print('Video already downloaded, Skipping')
            return

        print('Downloading video')

        yt = YouTube(self.url)

        vid_candidates = yt.streams.filter(
            progressive=True,
            file_extension='mp4'
        ).order_by('resolution').desc().first()

        print('Candidates found')

        if vid_candidates is None:
            print("No video found 💀")
            return
        
        vid_candidates.download(
            output_path=mg_dir_path(self.id),
            filename='video.mp4',
        )

        print('Video downloaded')

    def process_frames(self, skip_if_exists: bool = True):
        '''
        Saves the frames to the data folder
        '''
        frames_path = mg_dir_path(self.id) + 'frames/'

        if skip_if_exists and os.path.exists(frames_path):
            print('Frames already processed, Skipping')
            return

        vid = self.get_video()

        print('Processing frames')

        if not os.path.exists(frames_path):
            os.makedirs(frames_path)

        frame_num = 0
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            frame_path = frames_path + str(frame_num) + '.jpg'
            # resizing image to have height of 256, keeping aspect ratio
            frame = cv2.resize(frame, (int(frame.shape[1] * 256 / frame.shape[0]), 256))
            cv2.imwrite(frame_path, frame)
            frame_num += 1

    def get_video(self):
        path = mg_dir_path(self.id) + 'video.mp4'

        if not os.path.exists(path):
            print('Video not found')
            self.download_video()

        print('Loading video', path)

        cap = cv2.VideoCapture(path)
        return cap

    def get_frame_count(self):
        frame_path = mg_dir_path(self.id) + 'frames/'
        if not os.path.exists(frame_path):
            return 0

        num_frames = len(os.listdir(frame_path))

        return num_frames

    def get_frame_label(self, frame_num: int) -> ALL_LABELS:
        if len(self.action_frames) == 0:
            return "unlabeled"
        prev_label = "unlabeled"
        print('Action frames:', self.action_frames)
        for (frame_num_inner, label) in self.action_frames:
            if frame_num < frame_num_inner:
                return prev_label
            prev_label = label
        return prev_label

    def get_frame(self, frame_num: int) -> Optional[torch.Tensor]:
        if frame_num >= self.get_frame_count():
            print('Frame not found')
            return None
        path = mg_dir_path(self.id) + 'frames/' + str(frame_num) + '.jpg'
        img = cv2.imread(path)

        # resizing so smallest side is 224
        if img.shape[0] < img.shape[1]:
            img = cv2.resize(img, (int(224 * img.shape[1] / img.shape[0]), 224))
        else:
            img = cv2.resize(img, (224, int(224 * img.shape[0] / img.shape[1])))

        img = center_crop(img, (224, 224))

        # preprocess image
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
        return img_tensor

    def get_datapoints(self, frames_per_datapoint: int):
        datapoints = []
        for i in range(self.get_frame_count() - frames_per_datapoint):
            dp = MgDatapoint(self, i, frames_per_datapoint)
            datapoints.append(dp)
        return datapoints


@dataclass
class MgDatapoint(Dataset):
    mg: MG
    starting_frame: int
    num_frames: int

    def load_item(self):
        frames: List[torch.Tensor] = []
        for i in range(self.num_frames):
            frame = self.mg.get_frame(self.starting_frame + i)
            if frame is None:
                print('Frame not found')
                raise IndexError
            frames.append(frame)
        label = self.mg.get_frame_label(self.starting_frame)
        print('Label from load_item:', label)
        for frame in frames:
            print('Frame shape:', frame.shape)
        # returning frames and label (index of ALL_LABELS)
        # all_label_index = ['moving', 'not_moving', 'inspection', 'scramble', 'unlabeled'].index(label)
        # all_label_index = list(ALL_LABELS).index(label)
        all_label_index = get_args(ALL_LABELS).index(label)
        print('All label index:', all_label_index)
        return torch.stack(frames), all_label_index
        
    def view(self, checkpoint=None):
        (frames, is_moving) = self.load_item()
        print(frames[0].shape)

        if checkpoint is not None:
            model = TurnClassifier(hidden_dim=1024, num_classes=2, enc_name="vit-base", num_frames=2)
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            with torch.no_grad():
                pred = model(frames)
                print('Prediction in view:', pred)

        print('Is moving:', is_moving)
        for frame in frames:
            # img_np = frame.numpy()
            # undoing preprocessing by permuting, converting to numpy, and converting to 0-255
            img_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imshow('frame', img_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class MGDataset(Dataset):
    def __init__(self, mgs: List[MG], frames_per_item: int = 3):
        self.frames_per_item = frames_per_item
        
        self.dps: List[MgDatapoint] = []
        for mg in mgs:
            self.dps.extend(mg.get_datapoints(frames_per_item))

    def __len__(self):
        return len(self.dps)

    def __getitem__(self, idx: int):
        dp = self.get_data_point(idx)
        dp.mg.print()
        return dp.load_item()

    def get_data_point(self, idx: int):
        return self.dps[idx]

class DataPipeline:
    def download_all_videos(self):
        mgs = self.get_all_mgs()
        for mg in mgs:
            mg.download_video()

    def process_all_frames(self):
        mgs = self.get_all_mgs()
        for mg in mgs:
            mg.process_frames()

    def get_count(self):
        return len(os.listdir(mg_path))

    def get_dataset(self, split: Literal['train', 'test', 'both'] = 'both'):
        mgs = self.get_all_mgs()
        if split == 'train':
            mgs = list(filter(lambda mg: not mg.is_test, mgs))
        elif split == 'test':
            mgs = list(filter(lambda mg: mg.is_test, mgs))

        return MGDataset(mgs)

    def get_all_mgs(self):
        mgs = os.listdir(mg_path)
        all_mgs: List[MG] = []

        for mg in mgs:
            mg = MG.get_from_fs(int(mg))

            if mg is None:
                continue

            all_mgs.append(mg)

        return all_mgs

    def get_by_index(self, idx: int):
        mgs = self.get_all_mgs()
        print('Mgs found:', len(mgs))
        if idx > len(mgs):
            print('MG Index out of bounds')
            raise IndexError

        print('Returning mg', idx)
        for mg in mgs:
            mg.print()

        mg = mgs[idx]
        return mg

    def download_all(self, amount: int = 100):
        solves = []
        for i in range(1, amount + 1):
            solve = self.download_by_web_id(i)
            if solve is not None:
                print('Adding solve' + str(i))
                solves.append(solve)

        print('Solves found: ', len(solves))

        # merge solves with the same url
        solve_dict = {}
        
        for solve in solves:
            if solve.url in solve_dict:
                solve_dict[solve.url].moves.extend(solve.moves)
            else:
                solve_dict[solve.url] = MG(solve.id, solve.web_id, solve.url, solve.moves)
        
        merged_solves = list(solve_dict.values())

        print('Merged solves found: ', len(merged_solves))
        
        good_solves = []
        # Filter videos that dont have good videos
        for solve in merged_solves:
            pytube = YouTube(solve.url)
            try:
                pytube.check_availability()
            except:
                print("Video not available")
                continue
            if pytube.length > 10 * 60:
                print("Video too long")
                continue
            good_solves.append(solve)

        # Update ids
        for i, solve in enumerate(good_solves):
            solve.id = i

        print('Good solves found: ', len(good_solves))

        for solve in good_solves:
            solve.save_to_fs()

        return merged_solves

    def download_by_web_id(self, id: int):
        url = "https://reco.nz/solve/" + str(id)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        heading = soup.h1
        if heading is None or "3x3" not in heading.text:
            print("Not a 3x3 solve")
            return

        iframe = soup.iframe

        if iframe is None:
            print("No iframe found")
            return

        yt_url = iframe['src']

        # Check if video is available

        if type(yt_url) is not str:
            print("Multiple iframes found")
            return

        # map to list of hrefs
        urls: List[str] = list(map(lambda a: a.get("href"), soup.find_all('a')))
        # filter out urls with cubedb
        urls = list(filter(lambda u: "cubedb" in u, urls))

        # Maps to url object and get the alg url query
        algs = list(map(lambda u: parse_qs(urlparse(u).query).get("alg"), urls))

        if len(algs) == 0:
            print("No algs found")
            return

        val = algs[0]

        if val is None:
            print("No alg found 2")
            return

        moveGroups = val[0].split("\n")

        moves = []
        for grp in moveGroups:
            candidates = grp.split(" ")

            # remove all candidates after "//"
            commentStart = candidates.index("//")

            goodCandidates = candidates[:commentStart]
            moves += goodCandidates

        # Remove parenthesis
        moves = list(map(lambda m: m.replace("(", "").replace(")", ""), moves))

        solve = MG(id=id, web_id=id, url=yt_url, moves=moves)

        return solve


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--bs", type=int, default=32)
    # parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--checkpoint", type=str, default=None)
    # args = parser.parse_args()
        
    pipeline = DataPipeline()

    whole_dataset = pipeline.get_dataset('both')
    train_dataset = pipeline.get_dataset('train')
    test_dataset = pipeline.get_dataset('test')

    action = sys.argv[1]

    if action == "download":
        amount = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        pipeline.download_all(amount)

    elif action == "download_videos":
        pipeline.download_all_videos()

    elif action == "process_frames":
        pipeline.process_all_frames()
    
    elif action == "full_pipeline":
        amount = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        pipeline.download_all(amount)
        pipeline.download_all_videos()
        pipeline.process_all_frames()

    elif action == "get_item":
        idx = int(sys.argv[2])
        item = whole_dataset.get_data_point(idx)
        if item is not None:
            # if args.checkpoint is not None:
            if len(sys.argv) > 3:
                item.view(sys.argv[3]) # checkpoint
            else:
                item.view()
        else:
            print("Item not found")

    elif action == "mg":
        mg_id = int(sys.argv[2])

        mg = MG.get_from_fs(mg_id)
        if mg is None:
            print("MG not found")
            exit()

        action = sys.argv[3]
        if action == "video":
            mg.download_video()
        elif action == "process":
            mg.process_frames()
        elif action == "print":
            mg.print()
        else:
            print("No action found")
            exit()

    else:
        print("No action found")
        exit()


