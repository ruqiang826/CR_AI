# CR_AI
AI for clash royale

1. using logitech camera as eye.   
2. playing clash royale in genymotion, ubuntu 14.04.   
  the original solution is, play CR in ipad, and airplay the screen to TV, use camera to capture TV screen. Ipad is for operating, and TV for capture. This solution is more complicated.
  I found genymotion is a good solution. 
  genymotion do not provide edition for ubuntu 14.04, but you can update some libs for installation.
  I found genymotion sometimes can not start the emulation of android, virtualbox complained " eth0 is not configured correctly, hardware opengl is disabled". I fix it for a long time and failed. I restart the computer, it worked again. WTF.
  Further, you can even not use a camera for input, just capture screen from genymotion or virtualbox. But, I want a real eye.

3. For operation, the original solution is a mechanical arm. I searched for a lot, some startups have product to do that, like dobot or 7bot. But the arms are expensive on one hand, and on the other hand, use a mechanical arm is difficult for a AI. It will have so many things to do with the arm and ML algorithms.
  Cause I use genymotion insted of ipad, I will use software mouse operation instead of mechanical arms. Maybe I will use arms in the next edition.

4. The first step is to label the image. I use the git repository "labelImg",which use PASCAL VOC format.



## running
1. for simulator, run "genymotion". if the simulator run fail, restart the computer and run genymotion as soon as the computer start up. 
  it may because the port already being used by other program.
2. start clash royale.
3. use test_camera_video.py to adjust camera. use test_camera.py to capture image.

## model training
1. train a model for king and eking(enemy king). This two unit is easy to detect and require less labeled images.
2. train the second model for the six buildings(king and arena). use the previous model to label king tower, it can save some manual work.
 
