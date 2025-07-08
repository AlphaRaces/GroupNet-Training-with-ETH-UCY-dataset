import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class TrajectoryDataset(Dataset):
    
    def __init__(self, directories, l, Num_agents):
        self.l = l                            # length of the trajectory (you have to split l = h + f)
        self.Num_agents = Num_agents          # Maximum number of agents
        self.data = []
        self.angles = np.arange(0, 360, 30)  
        
        for directory in directories:
            file_path = os.path.join(directory, 'obsmat.txt')
            print(f'Loading file: {file_path}')
            column_names = ['frame', 'id', 'x', 'y', 'z', 'vx', 'vy', 'vz']
            df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
            
            # Convert frame and id to integers
            df['frame'] = df['frame'].astype(int)
            df['id'] = df['id'].astype(int)
            
            # Delete unnecessary columns
            df = df.drop(['y', 'vx', 'vy', 'vz'], axis=1)
            
            # Obtain unique frames
            frames = df['frame'].unique()
            
            for frame in frames:
                df_filtered = df[(df['frame'] >= frame - l * 10 + 1) & (df['frame'] <= frame)]
                
                # Filter agents with exactly h appearances
                agent_counts = df_filtered['id'].value_counts()
                valid_agents = agent_counts[agent_counts == l].index
                df_valid = df_filtered[df_filtered['id'].isin(valid_agents)]
                
                if len(valid_agents) > 0:
                    
                    '''
                    This array is creating to garantee that the tensor input is always the same size
                    -50 positions will create fictitious agents that will be ignored in the model
                    '''	
                    
                    history_array = np.full((self.Num_agents, self.l, 2), -50.0)
                    
                    for i, agent_id in enumerate(valid_agents[:self.Num_agents]):
                        agent_data = df_valid[df_valid['id'] == agent_id].sort_values(by='frame')
                        history_array[i, :, 0] = agent_data['x'].values
                        history_array[i, :, 1] = agent_data['z'].values
                    
                    # Add rotations
                    for angle in self.angles:
                        theta = np.radians(angle)
                        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                                    [np.sin(theta), np.cos(theta)]])
                        rotated_history = np.dot(history_array, rotation_matrix.T)
                        self.data.append(torch.tensor(rotated_history, dtype=torch.float32))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def main():
    
    directories = ['seq_eth', 'seq_hotel', 'zara01', 'zara02']
    l = 5   # large of the trajectory (you have to split l = h + f)
    Num_agents = 30 # Set the maximum number of agents

    dataset = TrajectoryDataset(directories, l, Num_agents)
    # Convertir directamente con np.stack
    data = np.stack([d.numpy() for d in dataset.data])

    # Dividir en train y test
    np.random.shuffle(data)
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    train_data = np.transpose(train_data, (0, 2, 1, 3))
    test_data = np.transpose(test_data, (0, 2, 1, 3))

    # Guardar
    np.save('train.npy', train_data)
    np.save('test.npy', test_data)

    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')




if __name__ == '__main__':
    main()


                
