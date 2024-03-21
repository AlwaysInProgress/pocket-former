import tkinter as tk
import cv2
import mg

class VideoPlayer:
    def __init__(self, window):
        self.window = window
        self.window.title("Video Player")

        self.video_path = ""
        self.cap = None
        self.frame_number = 0
        self.playingState = "paused"
        self.mgdataset = mg.MGDataset()
        self.mg_index = 1

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

        tk.Button(
            window,
            text="Next Video", 
            command=self.next_video
        ).pack()

        tk.Button(
            window,
            text="Prev Video", 
            command=self.prev_video
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

        self.is_moving_label = tk.Label(window, text="Is Moving: ")
        self.is_moving_label.pack()

        self.labeled_frames_label = tk.Label(window, text="Label Frames: ")
        self.labeled_frames_label.pack()

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.slider = tk.Scale(
            window, 
            from_=0, 
            to=0,
            orient=tk.HORIZONTAL,
            length=500,
            command=lambda _: self.on_slider_change()
        )
        self.slider.pack()

        self.window.bind("<Left>", self.backwards)
        self.window.bind("<Right>", self.forward)

        self.load_mg()
        self.loop()

    def load_mg(self):
        idx = self.mg_index
        mg = self.mgdataset.get_by_index(idx)
        print("Loading mg", idx)
        if mg is None:
            print("No mg found for index", idx)
            return
        self.mg = mg
        mg.print()
        self.cap = self.mg.get_video()
        self.max_frame: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.draw()


    def loop(self):
        if self.playingState == "fast_forwards":
            self.forward(10)

        elif self.playingState == "fast_backwards":
            self.backwards(10)

        self.window.after(5, self.loop)

    def draw(self):
        is_moving = self.mg.is_cube_moving(self.frame_number)
        self.is_moving_label.config(text="Is Moving: " + str(is_moving))

        labeled_frames = ','.join([str(frame) for frame in self.mg.action_frames])
        self.labeled_frames_label.config(text="Label Frames: " + labeled_frames)

        self.slider.config(to=self.max_frame)
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

    def next_video(self):
        self.mg_index += 1
        if self.mg_index >= self.mgdataset.get_count():
            self.mg_index = 0
        self.load_mg()

    def prev_video(self):
        self.mg_index -= 1
        if self.mg_index < 0:
            self.mg_index = self.mgdataset.get_count() - 1
        self.load_mg()

    def label(self):
        print("Label")
        self.mg.new_action(self.frame_number)
        self.mg.save_to_fs()
        self.draw()

    def remove_label(self):
        print("Remove Label")
        self.mg.remove_last_action()
        self.mg.save_to_fs()
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
