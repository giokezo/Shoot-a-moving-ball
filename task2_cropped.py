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
        self.k_m = k_m
        self.g = g
        self.dt = dt

    def calculate_trajectory_rk4(self, start_pos: Tuple[float, float], 
                               v0: Tuple[float, float], frames: int, total_frames: int) -> List[np.ndarray]:
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
            
            k2_v = acceleration(v + 0.5 * k1_v * self.dt)
            k2_p = v + 0.5 * k1_v * self.dt
            
            k3_v = acceleration(v + 0.5 * k2_v * self.dt)
            k3_p = v + 0.5 * k2_v * self.dt
            
            k4_v = acceleration(v + k3_v * self.dt)
            k4_p = v + k3_v * self.dt
            
            v += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * self.dt / 6.0
            pos += (k1_p + 2 * k2_p + 2 * k3_p + k4_p) * self.dt / 6.0
            
            positions.append(pos.copy())
        
        return positions

    def find_collision_trajectory(self, start_pos: Tuple[float, float], 
                                target_pos: Tuple[float, float], 
                                frames: int,
                                total_frames: int,
                                tolerance: float,
                                max_iters: int = 100) -> List[np.ndarray]:
        v = np.array([10.0, 10.0])  # Initial guess
        delta = 1e-2

        for iteration in range(max_iters):
            positions = self.calculate_trajectory_rk4(start_pos, tuple(v), frames, total_frames)
            final_pos = positions[-1]
            error_vector = np.array(target_pos) - final_pos
            error = np.linalg.norm(error_vector)

            if error < tolerance:
                return positions

            # Estimate Jacobian matrix numerically
            J = np.zeros((2, 2))
            for i in range(2):
                v_perturbed = v.copy()
                v_perturbed[i] += delta
                perturbed_positions = self.calculate_trajectory_rk4(start_pos, tuple(v_perturbed), frames, total_frames)
                perturbed_final_pos = perturbed_positions[-1]
                J[:, i] = (perturbed_final_pos - final_pos) / delta

            try:
                delta_v = np.linalg.solve(J, error_vector)
                v += delta_v * 0.5  # Add damping factor
            except np.linalg.LinAlgError:
                v = np.array([random.uniform(-20, 20), random.uniform(-20, 20)])
                continue

        return self.calculate_trajectory_rk4(start_pos, tuple(v), frames, total_frames)

def animate_trajectory(video_path: str, trajectory: List[np.ndarray], radius: float):
    cap = cv2.VideoCapture(video_path)
    trail = []
    
    for pos in trajectory:
        ret, frame = cap.read()
        if not ret:
            break
            
        for past_pos in trail:
            cv2.circle(frame, (int(past_pos[0]), int(past_pos[1])), 2, (255,255,0), -1)
            
        cv2.circle(frame, (int(pos[0]), int(pos[1])), int(radius), (255, 0, 0), -1)
        trail.append(pos.copy())
        
        cv2.imshow("Trajectory Animation", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    cap.release()

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

def animate_predicted_trajectory(video_path: str, original_trajectory: List[np.ndarray], 
                               predicted_trajectory: List[np.ndarray], 
                               collision_trajectory: List[np.ndarray],
                               radius: float):
    """
    Animate original trajectory, predicted trajectory, and collision trajectory with trails.
    Green: Original ball trajectory
    Red: Predicted future positions
    Blue: Colliding ball
    Yellow: Trails
    """
    cap = cv2.VideoCapture(video_path)
    last_frame = None
    
    # Initialize trails
    original_trail = []
    collision_trail = []
    
    # First, play through the original video frames
    for i, pos in enumerate(original_trajectory):
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()
        
        # Draw original ball trail
        original_trail.append(pos)
        for past_pos in original_trail:
            cv2.circle(frame, (int(past_pos[0]), int(past_pos[1])), 2, (255, 255, 0), -1)
        
        # Draw original ball position
        cv2.circle(frame, (int(pos[0]), int(pos[1])), int(radius), (0, 255, 0), -1)
        
        # Draw collision trajectory trail and position
        if i < len(collision_trajectory):
            collision_pos = collision_trajectory[i]
            collision_trail.append(collision_pos)
            
            # Draw collision trail
            for past_pos in collision_trail:
                cv2.circle(frame, (int(past_pos[0]), int(past_pos[1])), 2, (255, 255, 0), -1)
            
            # Draw current collision ball position
            cv2.circle(frame, (int(collision_pos[0]), int(collision_pos[1])), 
                      int(radius), (255, 0, 0), -1)
        
        cv2.imshow("Complete Trajectory", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    # Initialize predicted trail
    predicted_trail = []
    
    # Then, show predicted frames with all trajectories
    if last_frame is not None:
        for i, pred_pos in enumerate(predicted_trajectory):
            frame = last_frame.copy()
            
            # Draw original trajectory trail
            for past_pos in original_trail:
                cv2.circle(frame, (int(past_pos[0]), int(past_pos[1])), 2, (255, 255, 0), -1)
            
            # Draw predicted trail and current position
            predicted_trail.append(pred_pos)
            for past_pos in predicted_trail:
                cv2.circle(frame, (int(past_pos[0]), int(past_pos[1])), 2, (255, 255, 0), -1)
            cv2.circle(frame, (int(pred_pos[0]), int(pred_pos[1])), 
                      int(radius), (0, 0, 255), -1)
            
            # Draw collision trajectory trail and position
            collision_idx = i + len(original_trajectory)
            if collision_idx < len(collision_trajectory):
                collision_pos = collision_trajectory[collision_idx]
                collision_trail.append(collision_pos)
                
                # Draw collision trail
                for past_pos in collision_trail:
                    cv2.circle(frame, (int(past_pos[0]), int(past_pos[1])), 2, (255, 255, 0), -1)
                
                # Draw current collision ball position
                cv2.circle(frame, (int(collision_pos[0]), int(collision_pos[1])), 
                          int(radius), (255, 0, 0), -1)
            
            cv2.imshow("Complete Trajectory", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    cap.release()

def main_cropped():
    video_path = "Final_Project_NP_Giorgi_Kezevadze\Task2/video_ball_test_cropped.mp4"
    k_m = 0.0001
    g = -9.8

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    detector = BallDetector()
    simulator = TrajectorySimulator(k_m=k_m, g=g)

    background = detector.compute_background(video_path)
    trajectory_data = detector.detect_ball_positions(video_path, background)

    # Calculate initial velocity for prediction
    last_positions = trajectory_data.positions[-2:]
    v0 = tuple(np.array(last_positions[1]) - np.array(last_positions[0]))
    n = 50  # number of added frames

    # Calculate predicted trajectory
    predicted_trajectory = simulator.calculate_trajectory_rk4(
        start_pos=tuple(last_positions[-1]),
        v0=v0,
        frames=10,
        total_frames=n
    )

    # Calculate collision trajectory
    t = random.randint(len(trajectory_data.positions), 
                      len(trajectory_data.positions) + len(predicted_trajectory))
    start_pos = (random.randint(0, width-1), random.randint(0, height-1))
    target_idx = min(t - len(trajectory_data.positions), len(predicted_trajectory) - 1)
    target_pos = predicted_trajectory[target_idx]
    r_of_shooting_ball = trajectory_data.radii[0]
    tolerance = trajectory_data.radii[0] + r_of_shooting_ball

    collision_trajectory = simulator.find_collision_trajectory(
        start_pos=start_pos,
        target_pos=target_pos,
        frames=10,
        total_frames=t,
        tolerance=tolerance
    )

    # Animate all trajectories together
    animate_predicted_trajectory(
        video_path,
        [tuple(pos) for pos in trajectory_data.positions],
        predicted_trajectory,
        collision_trajectory,
        trajectory_data.radii[0]
    )

if __name__ == "__main__":
    main_cropped()