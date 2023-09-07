# How to reproduce the experiments
This model has been trained to predict delay as a performance metric. In this directory,
you will find the needed files to train/validate and predict the metrics.

### Dependencies

**Recommended: Python 3.7**

Please, ensure you use Python 3.7. Otherwise, we do not guarantee the correct installation of dependencies.

You can install all the dependencies by running the following commands.
```
pip install -r requirements.txt
```

## Download the data
You can download the datasets for this particular experiment here:
- [Datasets](https://github.com/BNN-UPC/NetworkModelingDatasets/tree/master/datasets_v4)

Otherwise, you can download the datasets using the following command:
```
wget -O nsfnet.zip https://bnn.upc.edu/download/dataset-v4-nsfnet/
wget -O geant2.zip https://bnn.upc.edu/download/dataset-v4-geant2/
wget -O gbn.zip https://bnn.upc.edu/download/dataset-v4-gbn/
wget -O topology_zoo.zip https://bnn.upc.edu/download/dataset-v4-topology-zoo/
wget -O abilene.zip https://bnn.upc.edu/download/dataset-v4-abilene-sndlib/
wget -O nobel.zip https://bnn.upc.edu/download/dataset-v4-nobel-sndlib/
wget -O geant.zip https://bnn.upc.edu/download/dataset-v4-geant-sndlib/
```

## Execute the code
Once you have downloaded the datasets, you can train the model by running the following command:
```
python main.py
```

**Note**: If you want to change the dataset, you can change the `train` and `test` directory variable in the `config.ini` file.