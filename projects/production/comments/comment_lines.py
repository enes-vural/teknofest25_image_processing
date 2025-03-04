
    #cv2.imshow("Targets",hsv_result)
    #cv2.imshow("Gray",gray_result)
    #cv2.imshow("Edge-2s",edges)
    # cv2.imshow("Result", result_with_white)



    #--------- BLUR ---------- 
    # blurred_frame = gray_frame
    #blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    #blurred_frame = cv2.bilateralFilter(gray_frame,9,75,75)
    # blurred_frame = cv2.medianBlur(gray_frame,ksize=5)
    #blurred_frame = cv2.erode(blurred_frame,kernel=np.ones((5,5),dtype=np.uint8),iterations=3)
    #blurred_frame = cv2.dilate(blurred_frame,kernel=np.ones((5,5),dtype=np.uint8),iterations=3)
    #blurred_frame = cv2.morphologyEx(blurred_frame,cv2.MORPH_OPEN,kernel=np.ones((3,3),dtype=np.uint8),iterations=1)
    # blurred_frame = cv2.morphologyEx(blurred_frame,cv2.MORPH_CLOSE,kernel=np.ones((3,3),dtype=np.uint8),iterations=2)

    #equalized_frame = cv2.equalizeHist(blurred_frame) (detayları manyak belirtiyor paraziti arttırıyor).

    #--------- DILATE ---------
    # dilateKernel = np.ones((3,3),dtype=np.uint8)
    # blurred_frame = cv2.dilate(blurred_frame,dilateKernel,iterations=5)

    #--------- THRESHOLD ----------
    #static threshold
    # _, otsu_thresh_val = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #adaptive threshold
    # adaptive_threshold = cv2.adaptiveThreshold(blurred_frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,9)

    #detects the shape's edges with canny function

    # combined_threshold = cv2.bitwise_and(adaptive_threshold,blurred_frame)
    # blurred_frame = combined_threshold