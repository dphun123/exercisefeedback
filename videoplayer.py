# implemented by ai
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

class ExerciseVideoPlayer:
    def __init__(self, frames, reps):
        """
        Initialize the video player with frames and rep information
        
        :param frames: List of video frames
        :param reps: List of tuples (start_frame, bottom_frame, end_frame) for each rep
        """
        self.frames = frames
        self.reps = reps
        
        # Current state tracking
        self.current_rep = 0
        self.current_frame = reps[0][0]
        self.is_playing = False
        self.animation = None
        
        # Set up the figure and axes
        self.fig, (self.ax_main, self.ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), 
                                                                gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(bottom=0.2, top=0.95, hspace=0.3)
        
        # Main video display
        self.img_main = self.ax_main.imshow(frames[self.current_frame])
        self.ax_main.set_title(f"Exercise Rep {self.current_rep + 1}")
        self.ax_main.axis('off')
        
        # Bottom frame display
        self.img_bottom = self.ax_bottom.imshow(frames[self.reps[self.current_rep][1]])
        self.ax_bottom.set_title("Bottom Frame")
        self.ax_bottom.axis('off')
        
        # Play/Pause button
        self.ax_play = plt.axes([0.4, 0.05, 0.2, 0.05])
        self.button_play = Button(self.ax_play, 'Play')
        self.button_play.on_clicked(self.toggle_play)
        
        # Previous Rep button
        self.ax_prev = plt.axes([0.1, 0.4, 0.05, 0.2])
        self.button_prev = Button(self.ax_prev, '←')
        self.button_prev.on_clicked(self.prev_rep)
        
        # Next Rep button
        self.ax_next = plt.axes([0.85, 0.4, 0.05, 0.2])
        self.button_next = Button(self.ax_next, '→')
        self.button_next.on_clicked(self.next_rep)

    def update_frame(self, val=None):
        """Update the displayed frame"""
        # Ensure we're within the current rep's frame range
        start, bottom, end = self.reps[self.current_rep]
        
        # If val is None, use current_frame, otherwise use the slider value
        frame_num = int(self.slider.val) if val is not None else self.current_frame
        
        # Clamp frame number to current rep's range
        frame_num = start
        
        # Update main frame
        self.img_main.set_data(self.frames[frame_num])
        
        # Update bottom frame (use bottom frame of current rep)
        self.img_bottom.set_data(self.frames[bottom])
        
        # Update title
        self.ax_main.set_title(f"Exercise Rep {self.current_rep + 1}")
        
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
        start_frame, bottom_frame, end_frame = self.reps[self.current_rep]
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
            interval=5,
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
