import os
import sys
from dataclasses import dataclass
from typing import List
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import json
import cv2
from pytube import YouTube

@dataclass
class Solve:
    id: int
    url: str
    moves: List[str]

    def get_video_path(self): 
        path = 'data/videos/' + str(self.id) + '.mp4'
        path = os.path.join(os.path.dirname(__file__), path)
        return path

    def download_video(self):
        print('Downloading video')

        yt = YouTube(self.url)

        vid_canidates = yt.streams.filter(
            progressive=True,
            file_extension='mp4'
        ).order_by('resolution').desc().first()

        print('Canidates found')

        if vid_canidates is None:
            print("No video found 💀")
            return
        
        vid_canidates.download(
            output_path='data/videos',
            filename=str(self.id) + '.mp4',
        )

        print('Video downloaded')

    def get_video(self):
        path = self.get_video_path()

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
        canidates = grp.split(" ")

        # remove all canidates after "//"
        commentStart = canidates.index("//")

        goodCanidates = canidates[:commentStart]
        moves += goodCanidates

    # Remove parenthesis
    moves = list(map(lambda m: m.replace("(", "").replace(")", ""), moves))

    print(moves)

    solve = Solve(id=id, url=yt_url, moves=moves)

    return solve


def download_all():
    solves = []
    for i in range(1, 100):
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
            solve_dict[solve.url] = Solve(solve.id, solve.url, solve.moves)
    
    merged_solves = list(solve_dict.values())

    return merged_solves


def save_to_fs(solves: List[Solve]):
    with open('data/solves.json', 'w') as f:
        solvesDicts = list(map(lambda s: s.__dict__, solves))
        s = json.dumps(solvesDicts)
        f.write(s)

def get_from_fs():

    if not os.path.exists('data/solves.json'):
        print('No solves found')
        return []

    with open('data/solves.json', 'r') as f:
        data = f.read()
        solves = json.loads(data)
        return list(map(lambda s: Solve(**s), solves))

def print_solves(solves: List[Solve]):
    for solve in solves:
        print(solve.id)
        print(solve.url)
        print(solve.moves)

if __name__ == "__main__":

    action = sys.argv[1]

    if action == "download":
        solves = download_all()
        save_to_fs(solves)

    elif action == "print":
        solves = get_from_fs()
        print_solves(solves) 

    elif action == "solve":
        print("getting solves")
        solves = get_from_fs()

        solve_id = int(sys.argv[2])

        solve = solves[solve_id]

        action = sys.argv[3]
        if action == "download":
            solve.download_video()
        else:
            print("No action found")
            exit()

    else:
        print("No action found")
        exit()


