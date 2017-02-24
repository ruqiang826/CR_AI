
import cv2
import time

cap = cv2.VideoCapture(0)
ret1 = cap.set(3, 1280)
ret2 = cap.set(4, 720)
i = 0
print "set width and height", ret1, ret2

print "loop"
while(True):
  ret, frame = cap.read()

    # Our operations on the frame come here
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Display the resulting frame
  #cv2.imshow('frame',gray)
  #if cv2.waitKey(1) & 0xFF == ord('q'):
  #    break
  if i >= 1000:
      break
  cv2.imwrite("/tmp/img_%d.jpg" % i,frame)
  i += 1
  time.sleep(0.3)

  print "write %d image" % i

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
