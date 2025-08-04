import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class BallTrajectory:
    positions: np.ndarray
    frames: np.ndarray
    radii: List[float]
    processed_frames: List[np.ndarray]

class TrajectorySimulator:
    def __init__(self, k_m: float = 0.0001, g: float = -9.8, dt: float = 1):
        self.k_m = k_m  # Air resistance coefficient divided by mass
        self.g = g      # Gravitational acceleration
        self.dt = dt    # Time step for simulation

    def calculate_trajectory_rk4(self, start_pos: Tuple[float, float], 
                               v0: Tuple[float, float], frames: int, total_frames : int) -> List[np.ndarray]:
        """Calculate trajectory using 4th order Runge-Kutta method."""
        pos = np.array(start_pos, dtype=np.float32)
        v = np.array(v0, dtype=np.float32)
        positions = [np.array(start_pos)]
        self.dt = frames/total_frames

        def acceleration(v):
            speed = np.linalg.norm(v)
            return np.array([-self.k_m * v[0] * speed, 
                           -self.g - self.k_m * v[1] * speed])
        
        for _ in range(int(frames/self.dt)):
            k1_v = acceleration(v)
            k1_p = v
            
            k2_v = acceleration(v + 0.5 * k1_v)
            k2_p = v + 0.5 * k1_v
            
            k3_v = acceleration(v + 0.5 * k2_v)
            k3_p = v + 0.5 * k2_v
            
            k4_v = acceleration(v + k3_v)
            k4_p = v + k3_v
            
            v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * self.dt / 6.0
            pos += (k1_p + 2 * k2_p + 2 * k3_p + k4_p) * self.dt / 6.0
            
            positions.append(pos.copy())
        
        return positions

    def find_collision_trajectory(self, start_pos: Tuple[float, float], 
                                target_pos: Tuple[float, float], 
                                frames: int,
                                total_frames :int,
                                tolerance: float,
                                max_iters: int = 100) -> Tuple[List[np.ndarray], float, float]:
        """Find initial velocity to hit target position using Newton's method."""
        v = np.array([0.0, 0.0])
        delta = 1e-2

        for iteration in range(max_iters):
            positions = self.calculate_trajectory_rk4(start_pos, v, frames, total_frames)
            error_vector = np.array(target_pos) - positions[-1]
            error = np.linalg.norm(error_vector)

            if error < tolerance:
                return positions

            # Estimate Jacobian matrix numerically
            J = np.zeros((2, 2))
            for i in range(2):
                v_perturbed = v.copy()
                v_perturbed[i] += delta
                perturbed_positions = self.calculate_trajectory_rk4(start_pos, v_perturbed, frames, total_frames)
                perturbed_error_vector = np.array(target_pos) - perturbed_positions[-1]
                J[:, i] = (perturbed_error_vector - error_vector) / delta

            try:
                delta_v = np.linalg.solve(J, error_vector)
                v += -delta_v
            except np.linalg.LinAlgError:
                print(f"Warning: Singular matrix at iteration {iteration}")
                continue

        print(f"Warning: Newton's method did not converge. Last velocities: vx={v[0]}, vy={v[1]}")
        return self.calculate_trajectory_rk4(start_pos, v, frames)

class BallDetector:
    @staticmethod
    def detect_edges(frame: np.ndarray) -> np.ndarray:
        """Detect edges in frame using Canny edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)

    def compute_background(self, video_path: str) -> np.ndarray:
        """Compute background model from video frames."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ret:
            raise Exception("Failed to read video")
        
        background = np.zeros_like(self.detect_edges(frame), dtype=np.float32)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            edges = self.detect_edges(frame)
            background += edges.astype(np.float32)
        
        cap.release()
        return (background / frame_count).astype(np.uint8)

    def detect_ball_positions(self, video_path: str, background: np.ndarray) -> BallTrajectory:
        """Detect ball positions in video frames."""
        cap = cv2.VideoCapture(video_path)
        positions = []
        frames = []
        processed_frames = []
        radii = []

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            edges = self.detect_edges(frame)
            diff = cv2.absdiff(edges, background)
            
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            points = np.column_stack(np.where(thresh > 0))
            
            if len(points) > 0:
                clustering = DBSCAN(eps=5, min_samples=5).fit(points)
                labels = clustering.labels_
                
                unique_labels = set(labels) - {-1}  # Remove noise label
                if unique_labels:
                    largest_cluster = max(unique_labels, 
                                       key=lambda x: np.sum(labels == x))
                    
                    cluster_points = points[labels == largest_cluster]
                    centroid = np.mean(cluster_points, axis=0)
                    
                    cx, cy = centroid[1], centroid[0]
                    positions.append((cx, cy))
                    frames.append(frame_number)
                    radii.append(np.max(np.linalg.norm(cluster_points - centroid, axis=1)))
                    processed_frames.append(frame.copy())
            
            frame_number += 1
        
        cap.release()
        return BallTrajectory(
            np.array(positions),
            np.array(frames),
            radii,
            processed_frames
        )

def animate_trajectory(video_path: str, trajectory: List[np.ndarray], radius: float):
    """Animate the calculated trajectory on the video frames."""
    """ HERE ANIMATE THE SAME THING  BUT WITH APPENDED FRAMES OF DETERMINE_REST_OF_THE VIDEO FUNCTION"""
    cap = cv2.VideoCapture(video_path)
    trail = []
    
    for pos in trajectory:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw trail
        for past_pos in trail:
            cv2.circle(frame, (int(past_pos[0]), int(past_pos[1])), 2, (255,255,0), -1)
            
        # Draw current position
        cv2.circle(frame, (int(pos[0]), int(pos[1])), int(radius), (255, 0, 0), -1)
        trail.append(pos.copy())
        
        cv2.imshow("Trajectory Animation", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    cap.release()

def main_uncropped():
    # Configuration
    video_path = "Task2\slow_throw_and_fall.mp4"
    k_m = 0.0001  # Air resistance coefficient / mass
    g = -9.8      # Gravitational acceleration
    
    # Initialize video capture to get dimensions
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Initialize detector and simulator
    detector = BallDetector()
    simulator = TrajectorySimulator(k_m=k_m, g=g)
    
    # Detect ball positions
    background = detector.compute_background(video_path)
    trajectory_data = detector.detect_ball_positions(video_path, background)
    
    if len(trajectory_data.positions) == 0:
        print("No ball positions detected!")
        return
        
    # Select random frame and position
    t = random.randint(0, len(trajectory_data.positions) - 1)
    print(t)
    start_pos = (random.randint(0, width-1), random.randint(0, height-1))

    
    # Calculate collision trajectory
    target_pos = trajectory_data.positions[t]
    r_of_shooting_ball = trajectory_data.radii[t]
    tolerance = trajectory_data.radii[t] + r_of_shooting_ball
    
    trajectory = simulator.find_collision_trajectory(
        start_pos=start_pos,
        target_pos=target_pos,
        frames=10,
        total_frames = t,
        tolerance=tolerance
    )
    
    # Animate result
    animate_trajectory(video_path, trajectory, r_of_shooting_ball)

if __name__ == "__main__":
    main_uncropped()