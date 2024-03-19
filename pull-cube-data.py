from dataclasses import dataclass
from typing import List
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import json


@dataclass
class Solve:
    id: int
    url: str
    moves: List[str]


def getSolveDataById(id: int):
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


def save_all_solves():
    solves = []
    for i in range(1, 100):
        solve = getSolveDataById(i)
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

    # Save to json file
    with open('data/solves.json', 'w') as f:
        solvesDicts = list(map(lambda s: s.__dict__, merged_solves))
        s = json.dumps(solvesDicts)
        f.write(s)


save_all_solves()    














