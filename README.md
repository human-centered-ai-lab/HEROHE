# HEROHE
Grand Challenge HEROHE (HER2 on HE)

I. QuPath image preprocessing:

- Create new project in QuPath
- Import slides into Project
- Run groovy script "QuPath/ProjectDetection.groovy" for Project ("Automate" -> "Show Script Editor" -> "Run" -> "Run for Project") or just pull the file on the opened QuPath which will open the script    editor and then ("Run" -> "Run for Project")
- The output will be in the Project QuPath Folder

II. SQL Database Setup in Python
  TODO

III. Run classification 

test_nn.py instructions:

NOTE: SQL Database must be set up by this point otherwise the following will not work.

1. Set SQL database credentials:
  To set SQL credentials open test_nn.py and change the DATABASE= item at the bottom. 
  Here you can find the neccesary entries (User, Pw, Name)

2. navigate to /path/to/script/test_nn.py\
command line:\
python3 /path/to/script/test_nn.py <command> --arg1 --arg2 

- Commands : test/train

- Arguments:\
--logs path/to/logdir Location of Tensorboard logs and model checkpoints\
    If not set will create the folder logs_her2 in script root directory and save checkpoints there\
--subdir /path/to/subdir/ Location of the subdirectory to save submission file to /path/to/subdir/ \
    If not set will create folder submissions_her2 in script root directory and save CSV file there \
--filename The filname of the generated CSV file (default) \
    Default: Her2_test_results.csv

- Example Usage:\
without args:\
python3 /path/to/script/test_nn.py train\
python3 /path/to/script/test_nn.py test\
with args:\
python3 /path/to/script/test_nn.py train --logs path/to/logdir --subdir path/to/submission_dir/ \
python3 /path/to/script/test_nn.py test --logs path/to/logdir --subdir path/to/submission_dir/ 

- Testing:\
When testing the loads a checkpoint from the path specified under log_dir, like in python_scripts/logs_her2 here for the default case. 
If you want to specify a certain loaction and filename for the results you can do so otherwise when no arguments are set the 
filpath and name will be as stated above.

Training:\
TODO
