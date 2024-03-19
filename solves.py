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

@dataclass
class Solve:
    id: int
    web_id: int
    url: str
    moves: List[str]
    # each entry is a frame number of when the cube starts moving or stops moving
    action_frames: List[int] = field(default_factory=list)

    def get_dir_path(self):
        path = 'data/solves/' + str(self.id) + '/'
        path = os.path.join(os.path.dirname(__file__), path)

        if not os.path.exists(path):
            os.makedirs(path)

        return path

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
        print('Loading solve from fs', id)
        path = 'data/solves/' + str(id) + '/data.json'
        path = os.path.join(os.path.dirname(__file__), path)
        if not os.path.exists(path):
            print('Solve not found')
            return None
        with open(path, 'r') as f:
            data = f.read()
            return Solve(**json.loads(data))

    def save_to_fs(self):
        print('Saving solve to fs', self.id)
        with open(self.get_dir_path() + 'data.json', 'w') as f:
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
            output_path=self.get_dir_path(),
            filename='video.mp4',
        )

        print('Video downloaded')

    def process_frames(self):
        '''
        Saves the frames to the data folder
        '''
        vid = self.get_video()

        print('Processing frames')

        frames_path = self.get_dir_path() + 'frames/'

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
        path = self.get_dir_path() + 'video.mp4'

        if not os.path.exists(path):
            print('Video not found')
            self.download_video()

        print('Loading video', path)

        cap = cv2.VideoCapture(path)
        return cap

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

    solve = Solve(id=id, web_id=id, url=yt_url, moves=moves)

    return solve


class Solves:
    def download_all(self, amount: int = 100):
        solves = []
        for i in range(1, amount + 1):
            solve = download_one_by_id(i)
            if solve is not None:
                print('Adding solve' + str(i))
                solves.append(solve)

        # merge solves with the same url
        solve_dict = {}
        
        for solve in solves:
            if solve.url in solve_dict:
                solve_dict[solve.url].moves.extend(solve.moves)
            else:
                solve_dict[solve.url] = Solve(solve.id, solve.web_id, solve.url, solve.moves)
        
        merged_solves = list(solve_dict.values())

        # Update ids
        for i, solve in enumerate(merged_solves):
            solve.id = i

        for solve in merged_solves:
            solve.save_to_fs()

        return merged_solves

    def download_all_videos(self):
        solves = os.listdir('data/solves')
        for solve in solves:
            solve = Solve.get_from_fs(int(solve))
            if solve is not None:
                solve.download_video()

    def get_solves_count(self):
        return len(os.listdir('data/solves'))

    def get_solve_by_index(self, idx: int):
        solves = os.listdir('data/solves')

        if idx >= len(solves):
            print('Index out of range')
            return None

        solveId = int(solves[idx])

        return Solve.get_from_fs(solveId)


if __name__ == "__main__":
    solves = Solves()

    action = sys.argv[1]

    if action == "download":
        amount = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        solves.download_all(amount)

    if action == "download_videos":
        solves.download_all_videos()

    elif action == "solve":
        solve_id = int(sys.argv[2])

        solve = Solve.get_from_fs(solve_id)
        if solve is None:
            print("Solve not found")
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


