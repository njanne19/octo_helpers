# octo_helpers
Helper repository for interaction with the Octo generalist robot policy transformer network. 

## Collection of Notes and Tribal Knowledge

Here are some things that I've learned about the Octo model and how the dataset is formatted. It follows a standard, or at least accepts datsets of the standard, [RLDS](https://github.com/google-research/rlds)

Looking at a single batch example from the demonstration dataset, the keys are:
```
absolute_action_mask: (1, 7)
action: (1, 5, 7)
observation.image_primary: (1, 2, 256, 256, 3)
observation.image_wrist: (1, 2, 128, 128, 3)
observation.pad_mask: (1, 2)
observation.pad_mask_dict.image_primary: (1, 2)
observation.pad_mask_dict.image_wrist: (1, 2)
observation.pad_mask_dict.proprio: (1, 2)
observation.pad_mask_dict.timestep: (1, 2)
observation.proprio: (1, 2, 8)
observation.timestep: (1, 2)
task.image_primary: (1, 256, 256, 3)
task.image_wrist: (1, 128, 128, 3)
task.language_instruction.attention_mask: (1, 16)
task.language_instruction.input_ids: (1, 16)
task.pad_mask_dict.image_primary: (1,)
task.pad_mask_dict.image_wrist: (1,)
task.pad_mask_dict.language_instruction: (1,)
task.pad_mask_dict.proprio: (1,)
task.pad_mask_dict.timestep: (1,)
task.proprio: (1, 8)
task.timestep: (1,)
```
### Dimensioning
We're looking at a single batch here, the batch index typically takes the first dimension of all the data formats. However, you'll notice that for some data formats there is a second dimension, which is the time horizon window dimension. From what I can tell, the model by default uses a 2-step time horizon window for observations, and a 5-step time horizon window for actions. How this is described in the documentation is that at inference time, the model considers both the current and previous observation to make a prediction, and from that prediction, comes 5 consecutive actions to take. However, the documentation mentions that if you want to do a more standard rollout procedure, you can just apply the first action to your robot and iterate the predcition step. Further, there is a way to change this time horizon window size, although I haven't played enough around with it yet to know for sure. 

### Pad Mask 
The pad mask is the encompassing way in which the model knows whether or not to consider a given portion of the obvservation/task data. For example, if the robot you're working with does not have a wrist camera, then in this configuration, you would have a padmask value of `[False, False]` for `observation.padmask_dict.image_wrist`. However, much more is yet to be discovered. 