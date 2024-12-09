# implemented by ai
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

class ExerciseVideoPlayer:
    def __init__(self, exercise, frames, reps):
        """
        Initialize the video player with frames and rep information

        :param frames: List of video frames
        :param reps: List of tuples (start_frame, bottom_frame, top_frame, angles, feedback) for each rep
        """
        self.exercise = exercise
        self.frames = frames
        self.reps = reps

        # Current state tracking
        self.current_rep = 0
        self.current_frame = reps[0][0]
        self.is_playing = False
        self.animation = None

        # Set up the figure with a more balanced layout
        self.fig = plt.figure(figsize=(16, 9))
        
        # Create a grid specification with more balanced columns
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        # Left side - Main video
        self.ax_main = self.fig.add_subplot(gs[0:2, 0])
        self.img_main = self.ax_main.imshow(frames[self.reps[self.current_rep][2]])
        self.ax_main.set_title(f"{self.exercise} Rep {self.current_rep + 1}")
        self.ax_main.axis('off')

        # Right side - Bottom frame
        self.ax_bottom = self.fig.add_subplot(gs[0, 1])
        self.img_bottom = self.ax_bottom.imshow(frames[self.reps[self.current_rep][1]])
        self.ax_bottom.set_title("Bottom Frame")
        self.ax_bottom.axis('off')

        # Angles section (directly under bottom frame)
        self.ax_angles = self.fig.add_subplot(gs[1, 1])
        self.ax_angles.axis('off')
        
        # Convert angles to multi-line string
        angles_str = "\n".join([f"{k}: {v}" for k, v in self.reps[self.current_rep][3].items()])
        self.text_angles = self.ax_angles.text(0.5, 0.5, f"Angles:\n{angles_str}",
                                               horizontalalignment='center',
                                               verticalalignment='center',
                                               transform=self.ax_angles.transAxes,
                                               fontsize=9)

        # Feedback section (under angles)
        self.ax_feedback = self.fig.add_subplot(gs[2, 1])
        self.ax_feedback.axis('off')
        
        # Prepare feedback with potential bullet points
        feedback_str = "\n".join([f"• {line}" for line in self.reps[self.current_rep][4]])
        self.text_feedback = self.ax_feedback.text(0.5, 0.5, f"Feedback:\n{feedback_str}",
                                                   horizontalalignment='center',
                                                   verticalalignment='center',
                                                   transform=self.ax_feedback.transAxes,
                                                   fontsize=9,
                                                   wrap=True)

        # Navigation buttons - positioned below the main video
        self.ax_prev = plt.axes([0.2, 0.02, 0.1, 0.05])
        self.button_prev = Button(self.ax_prev, '← Prev')
        self.button_prev.on_clicked(self.prev_rep)

        self.ax_play = plt.axes([0.45, 0.02, 0.1, 0.05])
        self.button_play = Button(self.ax_play, 'Play')
        self.button_play.on_clicked(self.toggle_play)

        self.ax_next = plt.axes([0.7, 0.02, 0.1, 0.05])
        self.button_next = Button(self.ax_next, 'Next →')
        self.button_next.on_clicked(self.next_rep)

        plt.tight_layout()

    def update_frame(self, val=None):
        """Update the displayed frame"""
        # Ensure we're within the current rep's frame range
        start, bottom, end, angles, feedback = self.reps[self.current_rep]
        
        # If val is None, use current_frame, otherwise use the slider value
        frame_num = int(self.slider.val) if val is not None else self.current_frame
        
        # Clamp frame number to current rep's range
        frame_num = max(start, min(frame_num, end))
        
        # Update main frame
        self.img_main.set_data(self.frames[frame_num])
        
        # Update bottom frame (use bottom frame of current rep)
        self.img_bottom.set_data(self.frames[bottom])
        
        # Update title
        self.ax_main.set_title(f"Exercise Rep {self.current_rep + 1}")

        # Update angles
        angles_str = "\n".join([f"{k}: {v}" for k, v in self.reps[self.current_rep][3].items()])
        self.text_angles.set_text(f"Angles:\n{angles_str}")
        
        # Update feedback with bullet points
        feedback_str = "\n".join([f"• {line}" for line in self.reps[self.current_rep][4]])
        self.text_feedback.set_text(f"Feedback:\n{feedback_str}")
        
        # Redraw
        self.fig.canvas.draw_idle()
        
        # Update current frame
        self.current_frame = frame_num
    
    def next_rep(self, event=None):
        """Move to the next rep"""

        
        if self.current_rep < len(self.reps) - 1:
            self.current_rep += 1
            # Recreate animation for new rep
            self.update_frame()
    
    def prev_rep(self, event=None):
        """Move to the previous rep"""
        
        if self.current_rep > 0:
            self.current_rep -= 1
            # Recreate animation for new rep
            self.update_frame()
    
    def toggle_play(self, event=None):
        """Toggle play/pause of the animation"""
        self.button_play.label.set_text('Play')
        self._start_animation()
    
    def _start_animation(self):
        """Start the animation for the current rep"""
        start_frame, bottom_frame, end_frame, angles, feedback = self.reps[self.current_rep]
        self.current_frame = start_frame
        
        # Create the FuncAnimation instance to update the frames
        def animate_func(frame_num):
            if frame_num > end_frame:
                return
            
            self.img_main.set_data(self.frames[frame_num])
            return [self.img_main]
        
        self.animation = FuncAnimation(
            self.fig, 
            animate_func, 
            frames=range(start_frame, end_frame + 1),
            interval=10,
            blit=True,
            repeat=False   # Stop after one playthrough
        )
        
        # Force the animation to start
        plt.draw()
        
    def _animate(self, frame_num):
        """Animate the video frames"""
        self.current_frame = frame_num
        self.update_frame()
    
    def show(self):
        """Display the video player"""
        plt.show()
