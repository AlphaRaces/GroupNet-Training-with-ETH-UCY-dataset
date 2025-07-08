# GroupNet-Training-with-ETH-UCY-dataset

Based on the original GroupNet model repository [go here](https://github.com/MediaBrain-SJTU/GroupNet), we trained it with ETH-UCY data for subsequent use in a pedestrian tracking model. Trajectories generated with ET-UCY data were normalized and augmented with rotations to capture motion patterns in all directions. Because the number of agents is not constant over time, fictitious agents were created away from the scene to maintain this feature of the original model.
