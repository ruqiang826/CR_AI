import cv2
import time

cap = cv2.VideoCapture(0)
ret1 = cap.set(3, 1280)
ret2 = cap.set(4, 720)

while(1):
  ret, frame = cap.read()
  cv2.imshow("capture", frame)
  time.sleep(0.3)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
 
cap.release()
cv2.destroyAllWindows()
