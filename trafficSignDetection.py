import cv2
import numpy as np

def process_video(path = "./MVI_1054.avi"):
	cap = cv2.VideoCapture(path)

	while (cap.isOpened()):
		# Take each frame
		ret, frame = cap.read()
		if ret == True:
			# Use Gaussian Blur to reduce high frequency noise
			# and allow us to focus on the structural objects inside the frame
			blurred = cv2.GaussianBlur(frame, (3, 3), 0)
			cv2.imshow('gh',blurred)
            #blurred = cv2.medianBlur(frame, 5)
			
			# Convert BGR to HSV
			hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

			
			# Threshold the HSV image to get only red
			red1 = cv2.inRange(hsv, (0, 100, 100), (15, 255, 255))
			red2 = cv2.inRange(hsv, (160, 100, 120), (180, 255, 255))
			red_mask = cv2.add(red1, red2)

			# Threshold the HSV image to get only blue
			blue_mask = cv2.inRange(hsv, (100, 120, 100), (120, 255, 255))

			mask = cv2.add(red_mask, blue_mask)

			# Erode to reduce noise and dilate to focus
			mask = cv2.erode(mask, None, iterations = 1)
			mask = cv2.dilate(mask, None, iterations = 3)

			# Find contours in the mask
			cnts = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]

			# Proceed if at least one contour was found
			if len(cnts) > 0:
				# Draw all contours and fill the contour interiors -> mask
				cv2.drawContours(image = mask, contours = cnts, contourIdx = -1, color = 255, thickness = -1)
				mask = cv2.dilate(mask, None, iterations = 5)
				mask = cv2.erode(mask, None, iterations = 5)

			# Draw a rectangle outside each contour
			cnts = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]
			for each in cnts:
				x, y, w, h = cv2.boundingRect(each)
				if w > 10 and h > 10 and float(h)/w > 0.9 and float(h)/w < 1.5:
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
			#result = cv2.bitwise_and(frame, frame, mask)
			
			cv2.imshow("frame", frame)
			#cv2.imshow("hsv", hsv)
			cv2.imshow("mask", mask)
			#cv2.imshow("result", result)

			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else:
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	process_video("./MVI_1054.avi")