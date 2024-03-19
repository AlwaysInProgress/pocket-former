import tkinter as tk
from tkinter import filedialog
import cv2
import solves
import random


class VideoPlayer:
    def __init__(self, window):
        self.window = window
        self.window.title("Video Player")

        self.video_path = ""
        self.cap = None
        self.current_frame = None

        tk.Button(
            window,
            text="Next Frame", 
            command=self.next_frame
        ).pack()

        tk.Button(
            window,
            text="Prev Frame", 
            command=self.previous_frame
        ).pack()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.solve_data = solves.get_from_fs()
        solve = self.solve_data[0]
        video = solve.get_video()

        self.cap = video
        self.update_frame()

        self.window.bind("<Left>", self.previous_frame)
        self.window.bind("<Right>", self.next_frame)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            print("Drawing frame")
            if ret:
                self.current_frame = frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                photo = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.window.mainloop()

    def previous_frame(self):
        print("previous_frame")
        if self.cap and self.cap.isOpened():
            current_frame_number = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame_number > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number - 2)
                self.update_frame()

    def next_frame(self):
        print("next_frame")
        if self.cap and self.cap.isOpened():
            self.update_frame()

if __name__ == "__main__":
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()
