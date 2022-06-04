# Lane Detection

This repository is a project about lane detection. Currently, the system relies entirely on traditional
computer vision techniques, including but not limited to masking, thresholding, and perspective transformation. The 
sole use of computer vision techniques makes the system vulnerable under extreme situations: a large portion of water
reflection, strong sunlight reflection, or extremely blurred lane lines. These problems can be alleviated by incorporating
deep learning techniques. In addition to segmenting driving lanes, the system also aims to obtain
the curve radius of the lane and the position offset relative to the lane center, which information 
can be helpful for more advanced topics like self-driving cars. A detailed tutorial, written by Addison Sears-Collins,
of a simpler but similar lane detection system can be found [here](https://automaticaddison.com/the-ultimate-guide-to-real-time-lane-detection-using-opencv/).
