import cv2
import threading
import constants.constants as const
import helper.init as init
import helper.threads as threads
import helper.print_helper as printh
import helper.cv2.cv2_helper as cv2_helper
import numpy as np

cv2h_instance = cv2_helper.Cv2Helper()
th_instance = threads.ObjectThreads()
init_instance = init.Initalize()

listener_thread = threading.Thread(target=th_instance.process_detected_shapes,daemon=True)
listener_thread.start()

cleaner_thread = threading.Thread(target=th_instance.remove_old_shapes,daemon=True)
cleaner_thread.start()

# init_instance.run()
# init_instance.checkVideoStartState()

camera = cv2.VideoCapture(0)

while True:

    if th_instance.detected_shapes:
        printh.shapesOnScreen()
        for shape_id, shape in th_instance.detected_shapes.items():
            printh.shapeFound(shape)
    else:
        printh.noneFound()
    cv2.waitKey(1)

    
    state,frame = camera.read()
    #--------------------------
    #state,frame = init_instance.video.read()


    if init_instance.ifVideoEnded(state):
        continue
    
    clean_frame = frame.copy()
    
    hsv_image = cv2h_instance.convertHsv(clean_frame)

    blue_mask = cv2.inRange(hsv_image, const.blue_lower, const.blue_upper)

    red_mask_lower = cv2.inRange(hsv_image, const.red_lower_1, const.red_upper_1)
    red_mask_upper = cv2.inRange(hsv_image, const.red_lower_2, const.red_upper_2)
    red_mask = cv2.bitwise_or(red_mask_lower, red_mask_upper)
    
    overlap_mask = cv2.bitwise_and(blue_mask, red_mask)

    final_red = cv2.subtract(red_mask, overlap_mask)
    final_blue = cv2.subtract(blue_mask, overlap_mask)

    final_mask = cv2h_instance.combineMaskOr(final_red,final_blue)
    hsv_result = cv2h_instance.combineMaskAnd(frame,frame,final_mask)
    
    gray_result = cv2h_instance.convertGray(hsv_result)
    gray_result =  cv2h_instance.medianBlur(gray_result,7)
    gray_result = cv2h_instance.erode(gray_result,3)
    gray_result = cv2h_instance.morphOpen(gray_result,3)


    #---------------- GRAY CONFIG ---------------
    clean2_frame = frame.copy()

    gray_frame = cv2h_instance.convertGray(clean2_frame)
    blurred_frame = cv2h_instance.gaussianBlur(gray_frame)

    equalized_frame = cv2h_instance.equalizeHist(blurred_frame)
    otsu_thresh = cv2h_instance.getOtsuThreshold(equalized_frame)
    otsu_thresh_val = cv2h_instance.getOtsuThresholdValue(equalized_frame)
    
    
    w_edges = cv2.Canny(blurred_frame, otsu_thresh_val * 0.5, otsu_thresh_val)
    fields = cv2.Canny(gray_result, 100, 200)


    weight_contours = cv2h_instance.findWContours(w_edges)
    field_contours = cv2h_instance.findFContours(fields)
    
    filled_frame = cv2h_instance.fillBlack(fields)

    cv2h_instance.drawContoursHSV(filled_frame,field_contours)
    

    full_screen_intense = cv2h_instance.get_full_screen_dominant_color(frame)


    if full_screen_intense == const.Color.BLUE.value:
        th_instance.addHexagonInside()
    elif full_screen_intense == const.Color.RED.value:
        th_instance.addTriangleInside()
        

    if weight_contours:

        for w_cont in weight_contours:
            w_epsilon = cv2h_instance.returnEpsilon(const.default_epsilon,w_cont)
            w_approx = cv2h_instance.returnApprox(w_cont,w_epsilon)
            w_area = cv2h_instance.returnArea(w_cont)

            if w_area <  cv2_helper.Cv2Helper.calculatePixelValue():
                continue

            w_edge_count = cv2h_instance.returnEdgeCount(w_approx)
            x,y,w,h = cv2h_instance.returnSizes(w_approx)

            if w_edge_count == 3 and cv2h_instance.getShapeColor(clean_frame, x, y, w, h) == const.Color.RED.value:
                cv2h_instance.startDrawContours(frame,w_approx)
                th_instance.addTriangleOutside(x,y)
                center_x, center_y = cv2h_instance.getCenterOfTriangle(w_approx)
                cv2h_instance.drawCenter(frame,center_x,center_y)
                cv2h_instance.drawText(frame,"Triangle Target",x,y)
                    
            elif w_edge_count == 4 and 0.80 <= (aspect_ratio := float(w) / h) <= 1.30:
                cv2h_instance.startDrawContours(frame,w_approx)
                shape_color = cv2h_instance.getShapeColor(clean_frame,x,y,w,h)
                th_instance.addSquareOutside(shape_color,x,y)
                cv2h_instance.drawText(frame,f"Weight {shape_color}",x,y)


    if field_contours:
        for cont in field_contours:
            epsilon = cv2h_instance.returnEpsilon(const.default_epsilon,cont)
            approx = cv2h_instance.returnApprox(cont,epsilon)
            area = cv2h_instance.returnArea(cont)

            if area < cv2_helper.Cv2Helper.calculatePixelValue():
                continue

            edge_count = cv2h_instance.returnEdgeCount(approx)
            x,y,w,h = cv2h_instance.returnSizes(approx)

            if edge_count==3 and cv2h_instance.getShapeColor(clean_frame,x,y,w,h) == const.Color.RED.value:
                cv2h_instance.startDrawContours(frame,approx)
                th_instance.addTriangleOutside(x,y)
                center_x, center_y = cv2h_instance.getCenterOfTriangle(approx)
                cv2h_instance.drawCenter(frame,center_x,center_y)
                cv2h_instance.drawText(frame,"Triangle Target",x,y)
                    
            elif edge_count == 4 and 0.80 <= (aspect_ratio := float(w) / h) <= 1.30:
                cv2h_instance.startDrawContours(frame,approx)
                shape_color = cv2h_instance.getShapeColor(clean_frame,x,y,w,h)
                th_instance.addSquareOutside(shape_color,x,y)
                cv2h_instance.drawText(frame,f"Weight {shape_color}",x,y)

        
            elif edge_count==6 and cv2h_instance.getShapeColor(clean_frame,x,y,w,h) == const.Color.BLUE.value:        
                sides = []
                for i in range(6):
                    # i point of hexagon
                    p1 = approx[i][0]
                    # i+1 point of hexagon
                    # %6 to wrap around if last point is 5 and second point is 0 in logic.
                    p2 = approx[(i + 1) % 6][0]
                    # Calculate the length of the side between two points.
                    side_length = np.linalg.norm(p1 - p2)
                    # Append the side length to the list
                    sides.append(side_length)

                # Get the average side length from the list 
                avg_side_length = np.mean(sides)

                # Calculate the deviation of each side from the average metric.
                side_deviation = [abs(side - avg_side_length) / avg_side_length for side in sides]

                # if all deviations are less than %15 of the average side length
                if all(deviation < 0.15 for deviation in side_deviation):
                    # get perimeter and area of the hexagon
                    peri = cv2.arcLength(approx, True)
                    area = cv2.contourArea(approx)
                    # Calculate the circularity of the hexagon
                    if peri != 0:
                        circularity = 4 * np.pi * (area / (peri * peri))
                    else:
                        circularity = 0

                    # if the circularity is between %70 and %95
                    # it is a hexagon
                    # in default hexagon circularity is between %70 and (%75 or %85)
                    # but in our case we are using %95 because it has more accuracy in my test :)
                    if 0.7 <= circularity <= 0.95:
                        # Done it's a hexagon !
                        #
                        cv2h_instance.startDrawContours(frame, approx)
                        th_instance.addHexagonOutside(x, y)
                        cv2h_instance.drawText(frame, "Hexagon Target", x, y)
                        cv2h_instance.drawCenter(frame, x + w / 2, y + h / 2)     

    else:
        pass

    cv2.imshow("Second Frame",w_edges)
    cv2.imshow("Final Mask",final_mask)
    cv2.imshow("Original",frame)

    if init_instance.initalizeKeyBinds():
        break




