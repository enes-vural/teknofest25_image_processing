
import cv2
import numpy as np
import time
from collections import deque


class ShapeDetector:
    """Class for detecting specific colored geometric shapes in video stream."""
    
    def __init__(self):
        # Color boundaries in HSV space (lower, upper)
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])  # Red wraps around HSV space
        self.red_upper2 = np.array([180, 255, 255])
        self.blue_lower = np.array([100, 100, 100])
        self.blue_upper = np.array([140, 255, 255])
        
        # Detection thresholds
        self.min_contour_area = 500  # Minimum area for a valid contour
        self.approx_polygon_epsilon = 0.04  # Epsilon value for polygon approximation
        
        # Parameters for shape verification
        self.triangle_angles = np.array([60, 60, 60])  # Expected angles for equilateral triangle
        self.angle_tolerance = 15  # Tolerance for angle comparison in degrees
        
        # FPS calculation
        self.fps_buffer = deque(maxlen=30)
        self.prev_frame_time = 0
        
        # Color intensity thresholds for "very close" detection
        self.color_intensity_threshold = 0.4  # 40% of the frame needs to be the color

    def preprocess_frame(self, frame):
        """Apply preprocessing to enhance detection reliability."""
        # Reduce noise while preserving edges
        processed = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Convert from BGR to HSV color space
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        # Apply adaptive histogram equalization to handle varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:,:,2] = clahe.apply(hsv[:,:,2])
        
        return hsv

    def create_color_masks(self, hsv_frame):
        """Create binary masks for red and blue colors."""
        # Create red masks (red wraps around HSV space)
        red_mask1 = cv2.inRange(hsv_frame, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv_frame, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Create blue mask
        blue_mask = cv2.inRange(hsv_frame, self.blue_lower, self.blue_upper)
        
        # Apply morphological operations to reduce noise and fill gaps
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        return red_mask, blue_mask

    def detect_shapes(self, frame):
        """Detect and identify shapes in the frame."""
        hsv_frame = self.preprocess_frame(frame)
        red_mask, blue_mask = self.create_color_masks(hsv_frame)
        
        # Check for "very close" cases using color intensity
        height, width = frame.shape[:2]
        frame_area = height * width
        
        red_close = np.sum(red_mask) / 255 > frame_area * self.color_intensity_threshold
        blue_close = np.sum(blue_mask) / 255 > frame_area * self.color_intensity_threshold
        
        if red_close:
            print("Red square/triangle is very close – inside the shape.")
        if blue_close:
            print("Blue square/hexagon is very close – inside the shape.")
        
        # Find contours in red mask
        red_shapes = self.process_contours(red_mask, frame, "red")
        
        # Find contours in blue mask
        blue_shapes = self.process_contours(blue_mask, frame, "blue")
        
        return red_shapes + blue_shapes

    def process_contours(self, mask, frame, color):
        """Process contours for a given color mask."""
        detected_shapes = []
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            # Filter out small noise contours
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            
            # Approximate contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_polygon_epsilon * peri, True)
            corners = len(approx)
            
            shape_type = None
            shape_color = color
            
            # Identify shape based on number of corners
            if color == "red" and corners == 3:
                # Verify it's actually a triangle by checking angles
                if self.verify_triangle(approx):
                    shape_type = "triangle"
            elif color == "red" and corners == 4:
                # Verify it's a square (not just any quadrilateral)
                if self.verify_square(approx):
                    shape_type = "square"
            elif color == "blue" and corners == 4:
                # Verify it's a square
                if self.verify_square(approx):
                    shape_type = "square"
            elif color == "blue" and 5 <= corners <= 7:  # Allow for some approximation error
                # Verify it's a hexagon
                if self.verify_hexagon(approx):
                    shape_type = "hexagon"
            
            # If a valid shape was detected, add it to the result
            if shape_type:
                detected_shapes.append((contour, shape_type, shape_color))
                
        return detected_shapes

    def verify_triangle(self, points):
        """Verify if the given points form a valid triangle."""
        # Check if we have 3 points
        if len(points) != 3:
            return False
        
        # Calculate angles
        angles = self.calculate_angles(points)
        
        # A triangle should have angles that sum to 180 degrees
        angle_sum = np.sum(angles)
        if not (175 <= angle_sum <= 185):  # Allow for small error
            return False
            
        return True
        
    def verify_square(self, points):
        """Verify if the given points form a valid square or rectangle."""
        # Check if we have 4 points
        if len(points) != 4:
            return False
        
        # Calculate angles
        angles = self.calculate_angles(points)
        
        # A square should have angles close to 90 degrees
        for angle in angles:
            if not (80 <= angle <= 100):  # Allow for some error
                return False
                
        # Check aspect ratio for squareness
        x, y, w, h = cv2.boundingRect(points)
        aspect_ratio = float(w) / h
        if not (0.8 <= aspect_ratio <= 1.2):  # Allow for some perspective distortion
            return False
            
        return True
    
    def verify_hexagon(self, points):
        """Verify if the given points approximate a hexagon."""
        # A true hexagon should have 6 points, but we allow 5-7 for approximation error
        if not (5 <= len(points) <= 7):
            return False
            
        # For a regular hexagon, internal angles should be 120 degrees
        angles = self.calculate_angles(points)
        for angle in angles:
            if not (100 <= angle <= 140):  # Allow for significant error due to approximation
                return False
                
        return True
        
    def calculate_angles(self, points):
        """Calculate internal angles of a polygon."""
        points = points.reshape(-1, 2)
        num_points = len(points)
        angles = []
        
        for i in range(num_points):
            p1 = points[i]
            p2 = points[(i + 1) % num_points]
            p3 = points[(i + 2) % num_points]
            
            # Create vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle using dot product
            dot = np.dot(v1, v2)
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            # Avoid division by zero
            if mag1 * mag2 == 0:
                angles.append(0)
                continue
                
            # Calculate angle in degrees
            cos_angle = dot / (mag1 * mag2)
            cos_angle = max(-1, min(cos_angle, 1))  # Ensure value is in valid range
            angle = np.degrees(np.arccos(cos_angle))
            
            angles.append(angle)
            
        return angles

    def draw_results(self, frame, shapes):
        """Draw detected shapes on the frame with labels."""
        for contour, shape_type, color in shapes:
            # Get contour center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
                
            # Draw contour
            color_bgr = (0, 0, 255) if color == "red" else (255, 0, 0)
            cv2.drawContours(frame, [contour], -1, color_bgr, 2)
            
            # Draw label
            label = f"{color} {shape_type}"
            cv2.putText(frame, label, (cx - 20, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
        return frame

    def calculate_fps(self):
        """Calculate and return the current FPS."""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = current_time
        self.fps_buffer.append(fps)
        return np.mean(self.fps_buffer)

    def process_frame(self, frame):
        """Process a single frame and return the annotated result."""
        # Calculate FPS for performance monitoring
        fps = self.calculate_fps()
        
        # Detect shapes
        shapes = self.detect_shapes(frame)
        
        # Draw results
        result_frame = self.draw_results(frame.copy(), shapes)
        
        # Add FPS counter
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        return result_frame


def main():
    """Main function to run the shape detection system."""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize shape detector
    detector = ShapeDetector()
    
    print("Shape Detection System Running...")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Process the frame
            result_frame = detector.process_frame(frame)
            
            # Display the result
            cv2.imshow('Colored Shape Detection', result_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Shape Detection System Stopped.")


if __name__ == "__main__":
    main()