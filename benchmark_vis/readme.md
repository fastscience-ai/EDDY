#### Create a conda environment with pythion3.8 and activate 
```
conda create -n dig python=3.8
conda activate dig
```
#### Install pytorch 1.11 with with other matching libraries
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```
#### make sure to install compatible scatter ,sparser and finally torch geometric with compatible hardware
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install torch-geometric==1.7.2
```
#### Install DIG 1.0.0  with steamlit and watchdog
```
pip install dive-into-graphs
pip install streamlit
pip install watchdog
```

After installation to run the app
```
streamlit run streamlit_app.py
```