# Grid-Distribution-Optimisation
Associated with Department of Energy Management Technologies at Technical University of Munich (TUM), Munich, Germany

The repository deals with a solution to the grid optimisation problem which aims to minimise the sum of line loading for the shown network
![image](https://github.com/user-attachments/assets/e60cafa8-e5d9-4fc1-82fd-ce499fa65ef6)

As seen in the image, there are 17 nodes which can be assigned a load and a (PV) generation curve. There are 17 load and generation profiles provided as shown in the following images.

![image](https://github.com/user-attachments/assets/16b276c0-0920-471e-9ffb-c043cead0104)
![image](https://github.com/user-attachments/assets/3e12be87-544d-4935-8b24-cacce62bd7a5)

In total there are (17!)^2 possibilities which make the problem impossible to solve based on iterative approach.

The study investigates Simulated Annealing optimisation approach based on literature review. A novel initialiser based on correlation matrix was used to provide quick initial guess for the Simulated Annealing algorithm to start with.
