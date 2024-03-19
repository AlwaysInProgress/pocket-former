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
            command=self.forward
        ).pack()

        tk.Button(
            window,
            text="Prev Frame", 
            command=self.backwards
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
            text="Pause", 
            command=self.pause
        ).pack()

        self.label_button = tk.Button(
            window,
            text="Label is Moving", 
            command=self.label
        )
        self.label_button.pack()

        self.remove_label_button = tk.Button(
            window,
            text="Remove Last Label", 
            command=self.remove_label
        )
        self.remove_label_button.pack()

        # Frame count text view
        self.frame_count_box = tk.Entry(window)
        self.frame_count_box.pack()

        self.is_moving_label = tk.Label(window, text="Is Moving: ")
        self.is_moving_label.pack()

        self.labeled_frames_label = tk.Label(window, text="Label Frames: ")
        self.labeled_frames_label.pack()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.solve_data = solves.get_from_fs()
        solve = self.solve_data[0]
        if solve is None:
            print("No solve found")
            return
        self.solve: solves.Solve = solve
        self.cap = self.solve.get_video()
        self.max_frame: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.slider = tk.Scale(
            window, 
            from_=0, 
            to=self.max_frame,
            orient=tk.HORIZONTAL,
            length=500,
            command=lambda _: self.on_slider_change()
        )
        self.slider.pack()

        self.draw()

        self.window.bind("<Left>", self.backwards)
        self.window.bind("<Right>", self.forward)



        self.loop()

    def loop(self):
        # print("Looping", self.playingState)
        if self.playingState == "fast_forwards":
            self.forward(10)

        elif self.playingState == "fast_backwards":
            self.backwards(10)

        self.window.after(5, self.loop)

    def draw(self):
        print("Draw")
        # Draw the frame number
        self.frame_count_box.delete(0, tk.END)
        self.frame_count_box.insert(0, str(self.frame_number))

        is_moving = self.solve.is_cube_moving(self.frame_number)
        self.is_moving_label.config(text="Is Moving: " + str(is_moving))

        labeled_frames = ','.join([str(frame) for frame in self.solve.action_frames])
        self.labeled_frames_label.config(text="Label Frames: " + labeled_frames)

        self.slider.set(self.frame_number)


        if is_moving:
            self.label_button.config(text="Add label: not moving")
        else:
            self.label_button.config(text="Add label: moving")

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

    def label(self):
        print("Label")
        self.solve.new_action(self.frame_number)
        solves.save_to_fs(self.solve_data)
        self.draw()

    def remove_label(self):
        print("Remove Label")
        self.solve.remove_last_action()
        solves.save_to_fs(self.solve_data)
        self.draw()

    def on_slider_change(self):
        self.frame_number = int(self.slider.get())
        self.draw()

    def fast_forwards(self):
        self.playingState = "fast_forwards"

    def fast_backwards(self):
        self.playingState = "fast_backwards"

    def pause(self):
        self.playingState = "paused"

    def backwards(self, amount = 1):
        if self.frame_number - amount < 0:
            self.frame_number = 0
        self.frame_number -= amount
        self.draw()

    def forward(self, amount = 1):
        if self.frame_number + amount > self.max_frame:
            self.frame_number = self.max_frame
        self.frame_number += amount
        self.draw()

if __name__ == "__main__":
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()
