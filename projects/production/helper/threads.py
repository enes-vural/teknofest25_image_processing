import queue
import time
import threading
import constants.constants as const
class ObjectThreads:
    #Queue for detected shapes
    shape_queue = queue.Queue()
    #Dictionary for detected shapes
    detected_shapes = {}

    def addHexagonInside(self):
        self.update_add_detected_shape(const.Shape.HEXAGON.value, const.Color.BLUE.value, (0,0))

    def addTriangleInside(self):
        self.update_add_detected_shape(const.Shape.TRIANGLE.value, const.Color.RED.value, (0,0))

    def addTriangleOutside(self,x,y):
        self.update_add_detected_shape(const.Shape.TRIANGLE.value, const.Color.RED.value, (x,y))

    def addHexagonOutside(self,x,y):
        self.update_add_detected_shape(const.Shape.HEXAGON.value, const.Color.BLUE.value, (x,y))
    
    def addSquareOutside(self,color,x,y):
        self.update_add_detected_shape(const.Shape.SQUARE.value, color, (x,y))
    
    def add_detected_shape_queue(self,shape_type, shape_color, position):
        shape_info = {
            "type":shape_type,
            "color":shape_color,
            "position":position,
            "time_stamp":time.time()
        }
        self.shape_queue.put(shape_info)

    def remove_old_shapes(self):
        while True:
            time.sleep(1)
            current_time = time.time()
            to_remove = [key for key, shape in self.detected_shapes.items() if current_time - shape["time_stamp"] > const.SHAPE_TIMEOUT]

            for key in to_remove:
                del self.detected_shapes[key]
            # cv2.waitKey(1)

    def update_add_detected_shape(self,shape_type, shape_color, shape_position):
        shape_id = f"{shape_type}_{shape_color}"
        shape_info = {
            "id":shape_id,
            "type":shape_type,
            "color":shape_color,
            "position":shape_position,
            "time_stamp":time.time()
        }
        self.detected_shapes[shape_id] = shape_info

    def process_detected_shapes(self):
        while True:
            shape_info = self.shape_queue.get()
            #print(f"Detected Objects {shape_info['type']}")
            self.update_add_detected_shape(
                shape_type=shape_info["type"],
                shape_color=shape_info["color"],
                shape_position=shape_info['position']
            )

    def __init__(self):
        self.threads = []