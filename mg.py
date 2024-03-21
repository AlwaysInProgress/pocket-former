import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import json
import cv2
from pytube import YouTube
from torch.utils.data import Dataset
import torch

mg_path = 'data/mg/'

def mg_dir_path(id: int):
    path = mg_path + str(id) + '/'
    path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

@dataclass
class MG:
    id: int # folder index
    web_id: int # index from website, we dont want to use this as the index
    url: str # video url
    moves: List[str] # list of notation moves
    # each entry is a frame number of when the cube starts moving or stops moving
    # even = start moving, odd = stop moving
    action_frames: List[int] = field(default_factory=list)

    def is_cube_moving(self, frame_num: int):
        is_moving = False
        for frame in self.action_frames:
            if frame_num < frame:
                break
            is_moving = not is_moving
        return is_moving

    def new_action(self, frame_num: int):
        self.action_frames.append(frame_num)

    def remove_last_action(self):
        self.action_frames.pop()

    @staticmethod
    def get_from_fs(id: int):
        print('Loading from fs', id)
        path = mg_dir_path(id) + 'data.json'
        if not os.path.exists(path):
            print('Mg not found')
            return None
        with open(path, 'r') as f:
            data = f.read()
            return MG(**json.loads(data))

    def save_to_fs(self):
        print('Saving to fs: ', self.id)
        with open(mg_dir_path(self.id) + 'data.json', 'w') as f:
            s = json.dumps(self.__dict__)
            f.write(s)
    
    def print(self):
        print("Id:", self.id)
        print("Url:", self.url)
        print("Moves:", self.moves)
        print("Action Frames:", self.action_frames)

    def download_video(self):
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

    def process_frames(self):
        '''
        Saves the frames to the data folder
        '''
        vid = self.get_video()

        print('Processing frames')

        frames_path = mg_dir_path(self.id) + 'frames/'

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
        return len(os.listdir(mg_dir_path(self.id) + 'frames/'))

    def get_frame(self, frame_num: int) -> Optional[torch.Tensor]:
        if frame_num >= self.get_frame_count():
            print('Frame not found')
            return None
        path = mg_dir_path(self.id) + 'frames/' + str(frame_num) + '.jpg'
        img = cv2.imread(path)
        # convert img to tensor
        img_tensor = torch.from_numpy(img)
        return img_tensor


def download_one_by_id(id: int):
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



@dataclass
class MGDatapoint(Dataset):
    frames: List[torch.Tensor]
    is_moving: bool

    def view(self):
        print('Is moving:', self.is_moving)
        for frame in self.frames:
            img_np = frame.numpy()
            cv2.imshow('frame', img_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



class MGDataset(Dataset):
    def __init__(self, frames_per_item: int = 3):
        self.frames_per_item = frames_per_item
        pass

    def __len__(self):
        mgs = self.get_all_mgs()
        item_count = 0
        for mg in mgs:
            item_count += mg.get_frame_count() - (self.frames_per_item - 1)
        return item_count

    def __getitem__(self, idx: int):
        mgs = self.get_all_mgs()
        for mg in mgs:
            frame_count = mg.get_frame_count()
            if idx < frame_count - (self.frames_per_item - 1):
                frames: List[torch.Tensor] = []
                for i in range(self.frames_per_item):
                    frames.append(mg.get_frame(idx + i))
                is_moving_label = mg.is_cube_moving(idx)
                return MGDatapoint(frames, is_moving_label)
            idx -= frame_count - (self.frames_per_item - 1)

    def download_all(self, amount: int = 100):
        solves = []
        for i in range(1, amount + 1):
            solve = download_one_by_id(i)
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

    def get_all_mgs(self):
        mgs = os.listdir(mg_path)
        all_mgs = []

        for mg in mgs:
            mg = MG.get_from_fs(int(mg))
            if mg is not None:
                all_mgs.append(mg)

        return all_mgs

    def get_by_index(self, idx: int):
        solves = os.listdir(mg_path)

        if idx >= len(solves):
            print('Index out of range')
            return None

        solveId = int(solves[idx])

        return MG.get_from_fs(solveId)


if __name__ == "__main__":
    dataset = MGDataset()

    action = sys.argv[1]

    if action == "download":
        amount = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        dataset.download_all(amount)

    elif action == "download_videos":
        dataset.download_all_videos()

    elif action == "get_item":
        idx = int(sys.argv[2])
        item = dataset.__getitem__(idx)
        if item is not None:
            item.view()
        else:
            print("Item not found")

    elif action == "mg":
        solve_id = int(sys.argv[2])

        solve = MG.get_from_fs(solve_id)
        if solve is None:
            print("MG not found")
            exit()

        action = sys.argv[3]
        if action == "download":
            solve.download_video()
        elif action == "process":
            solve.process_frames()
        elif action == "print":
            solve.print()
        else:
            print("No action found")
            exit()

    else:
        print("No action found")
        exit()


