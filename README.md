## Credits
 - All credits for the core code of the repo go to [walledka from Matterport](https://github.com/matterport/Mask_RCNN)

# INSTALLATION
 - Detailed instruction steps can be found in my [gist](https://gist.github.com/prerakmody/883459150bafc36f6dec4f46f0b13760)
 
## BASIC INSTALLATION (w/o cuda and nvidia) 
 - `sudo apt install python3-pip python3-dev`
 - `sudo pip3 install -r requirements.txt`

# RUNNING THE FLASK SERVER
 - Run `python3 index.py`
 - Alternatively run 'nohup python3 index.py &'
 - Port : 5000
 - [Postman Link](https://www.getpostman.com/collections/7feee5532e5678e364b1)

## MODEL WEIGHTS
 - You shall find .h5 model files in ./demo/model/libs/<foldername>
  - This is configurable in the src/predict.py file
 - in index.py you can use the CPU option instead of the GPU option for prediction
