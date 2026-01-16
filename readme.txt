I have a set of time series data tracking heart rate across exercise sessions of timed interval treadmill runs.
Each time series sample corresponds to one exercise session. Exercise sessions are executed in different days.
The data is available as set of csv files each with two columns: Time -- The relative time in seconds, HR (bpm) -- The heart rate measurement.

The exercise session has 4 intervals of rougly 4 minutes followed by 2 minutes of cool down.
In each interval there are 3 minutes of low intensity run followed by 1 minute of high intensity run.
The cool down period is a slow paced walk.

The time series sample starts roughly at the start of the exercise, but can end after the 2 minutes of cool down.
The time series sample contains roughly for peaks, which correspond to the high intensity interval minute, and 5 valleys, which correspond to the low intensity runs and the cool down.

I would like to align the data so that the peaks and the valleys are aligned.
Suggest a statistical model that I can fit to the data. The only transformation available here is time shifting of the samples.
Then, I would like to identify the peaks and the valley regions of each sample, as well as of the set as a whole. 
Then, I would like to get a measure of variability of the various parameters.

--- 

TODO:
move this to github.io blog post ?
setup github actions to automatically update the blog upon data upload?
how to deal with changing routines? E.g., once I plateau, I can start adding a new routine. Do I keep this as a standard measurement of fitness or do I just move to the new one?
* the goal is to measure progression in an interpretable way

