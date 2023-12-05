**1. Install Ubuntu 22.04**

**2. Install Nvidia Driver 515**
```
sudo ubuntu-drivers install nvidia-driver-515
```

**3. Python 3.8**
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8
sudo apt install python3.8-venv
```

**4. Install cuda 11.2.2**
```
sudo sh /home/erik/Downloads/cuda_11.2.2_460.32.03_linux.run --override
```

**5. Install libcudnn8**
```
sudo apt install ./libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
sudo apt install ./libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
```

**6. Add lines below to ~/.bashrc**
```
export PATH=$PATH:/usr/local/cuda-11.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64
```

**7. Make venv**
```
python3.8 -m venv plant_detection/plant_detection_venv
source plant_detection/plant_detection_venv/bin/activate
python -m pip install --upgrade pip
pip3 install -r ./plant_detection/src/requirements.txt

```


**8. Install gdal**
```
sudo apt install libgdal-dev gdal-bin
sudo apt install python3.8-dev
```

**9. Run gdalinfo --version, then get matching pygdal version**
```
pip install pygdal==${matching_pygdal_version}
```

**10. Ensure that ~/.keras/keras.json contains the following configuration**
```
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_last"
}
```

**11. Start server process to listen to requests from node.js server**
```
cd ./plant_detection/src
python server.py
```
