import numpy as np 

def create_dataset_from_arrays(images_primary, 
                               proprios, 
                               images_wrist = None, 
                               goal_type='image', 
                               goal_image_primary=None,
                               goal_image_wrist=None,
                               task_language_instruction=None, 
                               observation_horizon_window = 2
                               ):
    """ 
        Creates an RLDS style dataset to be used in performing inference 
        and fine-tuning on the Octo model. 
        
        Arguments (With shapes) : 
        
        N = Number of samples 
        H1 = Height of primary camera image
        W1 = Width of primary camera image
        H2 = Height of wrist camera image
        W2 = Width of wrist camera image
        C = Number of channels in the image
        J = Number of proprioceptive features (joints) 
        
        - images_primary : (N, H1, W1, C)
        - images_wrist : (N, H2, W2, C)
        - proprios : (N, J)
        - goal_type : str
        - goal_image_primary : (H1, W1, C)
        - goal_image_wrist : (H2, W2, C)
        - task_language_instruction : str
        
        Returns: 
        
        A length N-observation_horizon_window+1 list of dataset dictionaries, each dictionary containing
        the following keys: 
        
        - observation.image_primary : (1, observation_horizon_window, H1, W1, C)
        - observation.image_wrist : (1, observation_horizon_window, H2, W2, C)
        - observation.proprio : (1, observation_horizon_window, J)
        - observation.timestep: (1, observation_horizon_window)
        - task.image_primary : (1, H1, W1, C)
        - task.image_wrist : (1, H2, W2, C)
        - task.language_instruction : (1,) [str]
        
        Note taht the observation_horizon_window is the number of contextual timesteps included in a given data point. 
        This means that in the case that the observation_horizon_window is 2, the third data point in the dataset will have
        the following observations:
        
        - observation.image_primary[2] = [images_primary[1], images_primary[2]] along the second dimension
        - observation.image_wrist[2] = [images_wrist[1], images_wrist[2]]
        - observation.proprio[2] = [proprios[1], proprios[2]] 
        - observation.timestep[2] = [1, 2]
        NOTE: formatted as such to be compatible with the dimensions listed above. 
        
        Further, timesteps are simple 1-increments from 0 to N-1.
        
        If the goal_type is 'image', then task.image_primary and task.image_wrist will be the last 
        images of the sequence if they are not provided. 
        
        If the goal_type is 'language' then task.language_instruction will be the task_language_instruction
        
    """
    
    dataset = []
    N = images_primary.shape[0]
    observation_horizon = observation_horizon_window - 1
    
    for i in range(N): 
        if i < observation_horizon: 
            continue 
        
        
        observation = {}
        task = {}
        
        # Observation
        
        # If this is the first index
        
        observation['image_primary'] = np.expand_dims(images_primary[i-observation_horizon:i+1], axis=0)
        observation['image_wrist'] = np.expand_dims(images_wrist[i-observation_horizon:i+1], axis=0)
        observation['proprio'] = np.expand_dims(proprios[i-observation_horizon:i+1], axis=0)
        observation['timestep'] = np.expand_dims(np.arange(i-observation_horizon, i+1), axis=0)
        
        # Task
        if goal_type == 'image':
            
            if goal_image_primary is not None:
                task['image_primary'] = np.expand_dims(goal_image_primary, axis=0)
            else:
                task['image_primary'] = np.expand_dims(images_primary[N-1], axis=0)
                
            if goal_image_wrist is not None:
                task['image_wrist'] = np.expand_dims(goal_image_wrist, axis=0)
            else:
                task['image_wrist'] = np.expand_dims(images_wrist[N-1], axis=0)

        elif goal_type == 'language':
            task['language_instruction'] = task_language_instruction
        
        dataset.append({'observation': observation, 'task': task})
    
    return dataset