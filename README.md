# Lane Detection

This repository is a project about lane detection. Currently, the system relies completely on traditional
computer vision techniques, including but not limited to masking, thresholding, and perspective transformation. The 
sole use of computer vision techniques make the system vulnerable under extreme situations, like water
reflection, strong sunlight reflection, or blurred lane lines. These problems can be alleviated by incorporating
deep learning techniques. In addition to segmenting driving lane, the system also aims to obtaining
the curve radius of the lane and the position offset relative to the lane center, which information 
can be useful for more advanced topics like self-driving cars.
