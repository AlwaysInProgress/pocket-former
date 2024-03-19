import tkinter as tk
import cv2
import solves

class VideoPlayer:
    def __init__(self, window):
        self.window = window
        self.window.title("Video Player")

        self.video_path = ""
        self.cap = None
        self.frame_number = 0
        self.playingState = "paused"

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

        tk.Button(
            window,
            text="Fast Forwards", 
            command=self.fast_forwards
        ).pack()

        tk.Button(
                window,
                text="Fast Backwards", 
                command=self.fast_backwards
                        ).pack()
        tk.Button(
                window,
                text="Pasue", 
                command=self.pause
                ).pack()

        # Frame count text view
        self.frame_count_box = tk.Entry(window)
        self.frame_count_box.pack()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.solve_data = solves.get_from_fs()
        solve = self.solve_data[0]
        video = solve.get_video()

        self.cap = video
        self.draw()

        self.window.bind("<Left>", self.previous_frame)
        self.window.bind("<Right>", self.next_frame)

        self.loop()

    def loop(self):
        # print("Looping", self.playingState)
        if self.playingState == "fast_forwards":
            self.next_frame()

        elif self.playingState == "fast_backwards":
            self.previous_frame()

        self.window.after(10, self.loop)

    def draw(self):
        print("Draw")
        # Draw the frame number
        self.frame_count_box.delete(0, tk.END)
        self.frame_count_box.insert(0, str(self.frame_number))

        if not self.cap or not self.cap.isOpened():
            print("No video loaded")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number - 1)

        ret, frame = self.cap.read()

        if not ret:
            print("No frame found")
            return

        frame = frame[:,:,::-1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        photo = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
        print("Drawing frame", self.frame_number)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        self.canvas.pack()

    def fast_forwards(self):
        self.playingState = "fast_forwards"

    def fast_backwards(self):
        self.playingState = "fast_backwards"

    def pause(self):
        self.playingState = "paused"

    def previous_frame(self):
        self.frame_number -= 1
        self.draw()

    def next_frame(self):
        self.frame_number += 1
        self.draw()

if __name__ == "__main__":
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()
