# WindPower
Manual: 
Some packages will not be installed, and you will therefore need to create a venv (on vs code), or install them in your environment (if using colab or kaggle). Most likely one package you won't have preinstalled will be "pytorch lightning", and to install it you would od "pip install pytorch-lightning". 

CNN: 
Run the one_feature_pipeline_cnn.py (the cnn model), you should run "python3 one_feature_pipeline_cnn.py --data_path [insert path to data]". 

LSTM: 
Run the one_feature_pipeline.py (the LSTM model), by doing "python3 one_feature_pipeline.py --data_path [insert path to data]"

Neural ODE: 
For the neural ode, do "python3 neural_ode.py". To test on different datasets, go to line 258 and change the data path it refers to, and then run the code.