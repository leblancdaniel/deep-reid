# deep-reid

Includes code to test OSNet models for feature extraction.  Original repo 
here: https://github.com/KaiyangZhou/deep-person-reid

You will run the file `reid.py` after installing all the dependencies.  Note that 
you can change the model type (I've selected the very light OSNet_x0_25) and the image path.

## Demo of OSNet (feature extraction)

    # create environment
    cd deep-reid/
    conda create --name torchreid python=3.7
    conda activate torchreid

    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop

    # run example script
    python reid.py


