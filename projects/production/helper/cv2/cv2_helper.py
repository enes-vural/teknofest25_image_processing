import constants.constants as const
import numpy as np
import cv2

class Cv2Helper:
    
    def nothing():
        pass

    def calculatePixelValue() ->int:
        if const.isDeviceMac:
            return 7744
        else:
            return 1444
        
    def getCenterOfTriangle(self,approx):
        triangle_points = approx.reshape(-1,2)
        x1,y1 = triangle_points[0]
        x2,y2 = triangle_points[1]
        x3,y3 = triangle_points[2]

        center_x = (x1+x2+x3)/3
        center_y = (y1+y2+y3)/3

        return center_x,center_y
        
    
    def startDrawContours(self,frame,approx):
        cv2.drawContours(frame,[approx],0,(0,255,0),2)

    def drawCenter(self,frame,x,y):
        cv2.circle(frame,(int(x),int(y)),5,(255,0,0),-1)

    def drawText(self,frame,text,x,y):
        cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    def drawContoursHSV(self,filled_frame,field_contours):
        cv2.drawContours(filled_frame,field_contours,-1,(255,255,255),thickness=cv2.FILLED)

    def findWContours(self,frame):
        cont,_ = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        return cont
    
    def findFContours(self,frame):
        cont,_ = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return cont
    
    def fillBlack(self,field):
        return np.zeros_like(field)

    def returnSizes(self,approx):
        return cv2.boundingRect(approx)

    def returnEpsilon(self,value:float,cont):
        return value* cv2.arcLength(cont,True)
    
    def returnApprox(self,cont,epsilon):
        return cv2.approxPolyDP(cont,epsilon,True)
    
    def returnArea(self,cont):
        return cv2.contourArea(cont)
    
    def returnEdgeCount(self,approx):
        return len(approx)

    def convertHsv(self,frame):
        return cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    def convertGray(self,frame):
        return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    def medianBlur(self,frame,kernel):
        return cv2.medianBlur(frame,kernel)

    def gaussianBlur(self,frame):
        return cv2.GaussianBlur(frame,(5,5),0)
    
    def equalizeHist(self,frame):
        return cv2.equalizeHist(frame)
    
    def getOtsuThreshold(self,frame):
        _,thresh = cv2.threshold(frame,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def getOtsuThresholdValue(self,frame):
        otsu_thresh_val,_ = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu_thresh_val

    def erode(self,frame,iterations):
        return cv2.erode(frame,const.kernelErode,iterations)
    
    def morphOpen(self,frame,iterations):
        return cv2.morphologyEx(frame,cv2.MORPH_OPEN,const.kernelOpen,iterations)
    
    def combineMaskOr(self,mask1,mask2):
        return cv2.bitwise_or(mask1,mask2)
    
    def combineMaskAnd(self,src1,src2,mask):
        return cv2.bitwise_and(src1,src2,mask)

    def getShapeColor(self,clean_frame,x,y,w,h)->str:
        
        shape_dominant_index = None
        histograms = {}
        shape_dominant_color = {}

        crop_image = clean_frame[y:y+h,x:x+w]

        for i,color in enumerate(const.shape_colors):

            color_hist = cv2.calcHist([crop_image],[i],None,[256],[0,256])
            #histograms dictionary holds the color_hist values
            histograms[color] = color_hist

        for color in histograms:
            max_intensity = np.argmax(histograms[color])
            shape_dominant_color[color] = max_intensity
        shape_dominant_index = max(shape_dominant_color,key=shape_dominant_color.get)

        return const.color_names[shape_dominant_index]




    def get_full_screen_dominant_color(self,frame)->str:
        _frame = frame.copy()

        color_ratios = {}

        total_pixels = _frame.shape[0] * _frame.shape[1]

        b_channel, g_channel, r_channel = cv2.split(_frame)
        
        b_intensity = np.sum(b_channel) / (total_pixels * 255) * 100  # % cinsinden
        g_intensity = np.sum(g_channel) / (total_pixels * 255) * 100
        r_intensity = np.sum(r_channel) / (total_pixels * 255) * 100

        general_intensity = (b_intensity + g_intensity + r_intensity) / 3

        color_ratios['b'] = b_intensity
        color_ratios['g'] = g_intensity
        color_ratios['r'] = r_intensity

        dominant_colors = [color for color in color_ratios if color_ratios[color] > 30]
        
        if dominant_colors:
            dominant_color = dominant_colors[0]
            dominant_color_intensity = color_ratios[dominant_color]
            if abs(dominant_color_intensity - general_intensity >= 10):
                return const.color_names[dominant_colors[0]]

        return "No dominant color"
