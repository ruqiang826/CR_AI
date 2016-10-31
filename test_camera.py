
import cv2

cap = cv2.VideoCapture(0)
i = 0

while(True):
  ret, frame = cap.read()

    # Our operations on the frame come here
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Display the resulting frame
  cv2.imshow('frame',gray)
  #if cv2.waitKey(1) & 0xFF == ord('q'):
  #    break
  if i >= 10:
      break
  cv2.imwrite("filename_%d.jpg" % i,frame)
  i += 1
  print "write %d image" % i

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
