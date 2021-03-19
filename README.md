# Description
This code is focused on the application of a certain ML algorithms in a specific scenario in order to estimate the received signal strength (RSS) at differents point in that scenario, from the information provided by low-cost sensors whose measurements have a very low precise. The selected ML algorithm is Gaussian Processes for Regression (GPR) and the code is implemented in Python3.6. The theoretical background is based on the following project: "Recursive Estimation of Dynamic RSS Fields Based of Crowdsourcing and Gaussiand Processes" which authors are: Irene Santos, Juan José Murillo-Fuentes and Petar M. Djuric.

# Steps to execute the algorithm
1. First of all, install the necessary libraries in a virtual environment. For that, use the requirements.txt file uploaded on this project.
2. Then, examine the main file (static_GP.py), specially the lines from 40-59. These lines describe the flags to execute the different scenario developed for this project. Mainly, to execute the usual scenario to static field (the sensores are always in the same position) along with the time, you shall select the option '1' in the first position in line 58 and the option '1' in the third position. On the other hand, if you want to analyze the recursive scenario, you shall set the option '2' in the first position and the option '1' in the fourth position. The remaining paramaters can be set by default or you can change to study the different scenarios. 
3. Save the generated graphics and analize the result. 

# Code analysis
The developed code is described in file: code.pdf. This file includes a description about the classes and functions in the Python code, and their correspondence with the mathematical background presented in the paper mentioned previously.  
