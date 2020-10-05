# Does MAML really want feature reuse only?

This repository is the official implementation of "Does MAML really want feature reuse only?"
Our implementations are relied on [Torchmeta](https://github.com/tristandeleu/pytorch-meta). 

## Requirements

We run our code in the following environment using Anaconda.

- Python >= 3.5
- Pytorch == 1.4
- torchvision == 0.5

If you use Pytorch version above 1.5 (which is the latest version at this moment) and torchvision above 0.6, you may encounter problem. In that case, you are encouraged to change to the version in our environment.

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

If you want to train 4conv network in the paper, run this command:

```train
./run_4conv.sh
```

If you want to train ResNet-12 in the paper, run this command:

```train
./run_resnet.sh
```

If you want to see and change the arguments of training code, run this command:
```
python3 main.py --help
```

## Evaluation

To evaluate the model(s) and see the results, please refer to the `analysis.ipynb`


## Results

All results were reproduced by our group and reported as the average and standard deviation of the accuracies over 5x1000 tasks.

The values in parenthesis are the number of shots.

### 1. 5-Way k-shot test accuracy (%) of 4conv network on various benchmark dataset.

<table align="center" style="margin: 0px auto;">
    <tr>
        <td>Domain</td>
        <td colspan="2">General Domain</td>
        <td colspan="2">Specific Domain</td> 
    </tr>
    <tr>
        <td>Dataset</td>
        <td>miniImageNet</td>
        <td>tieredImageNet</td>
        <td>CUB</td>
        <td>Cars</td>
    </tr>
    <tr>
        <td>MAML(1)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?48.47 \pm 0.26" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?48.80 \pm 0.34" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?53.70 \pm 0.42" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?38.16 \pm 0.20" /></td>
    </tr>
    <tr>
        <td>BOIL(1)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?49.65 \pm 0.19" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?50.00 \pm 0.35" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?60.45 \pm 0.45" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?65.11 \pm 0.36" /></td>
    </tr>
    <tr>
        <td>MAML(5)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?60.36 \pm 0.25" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?64.27 \pm 0.27" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?65.11 \pm 0.07" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?45.36 \pm 0.23" /></td>
    </tr>
    <tr>
        <td>BOIL(5)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?65.32 \pm 0.34" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?69.64 \pm 0.20" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?74.12 \pm 0.24" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?65.70 \pm 0.17" /></td>
    </tr>
</table>


### 2. 5-Way k-shot test accuracy (%) of 4conv network on cross-domain adaptation.

<table align="center" style="margin: 0px auto;">
    <tr>
        <td>Adaptation</td>
        <td colspan="2">General to general</td>
        <td colspan="2">General to Specific</td> 
    </tr>
    <tr>
        <td>Meta-train</td>
        <td>tieredImageNet</td>
        <td>miniImageNet</td>
        <td>miniImageNet</td>
        <td>miniImageNet</td>
    </tr>
    <tr>
        <td>Meta-test</td>
        <td>miniImageNet</td>
        <td>tieredImageNet</td>
        <td>CUB</td>
        <td>Cars</td>
    </tr>
    <tr>
        <td>MAML(1)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?49.45 \pm 0.31" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?52.31 \pm 0.33" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?40.36 \pm 0.12" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?35.27 \pm 0.11" /></td>
    </tr>
    <tr>
        <td>BOIL(1)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?51.35 \pm 0.18" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?54.09 \pm 0.41" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?44.38 \pm 0.11" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?37.16 \pm 0.35" /></td>
    </tr>
    <tr>
        <td>MAML(5)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?65.31 \pm 0.12" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?64.88 \pm 0.28" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?51.34 \pm 0.24" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?44.29 \pm 0.28" /></td>
    </tr>
    <tr>
        <td>BOIL(5)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?70.76 \pm 0.14" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?68.97 \pm 0.24" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?60.11 \pm 0.32" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?50.92 \pm 0.22" /></td>
    </tr>
</table>

---

<table align="center" style="margin: 0px auto;">
    <tr>
        <td>Adaptation</td>
        <td colspan="2">Specific to general</td>
        <td colspan="2">Specific to Specific</td> 
    </tr>
    <tr>
        <td>Meta-train</td>
        <td>CUB</td>
        <td>CUB</td>
        <td>Cars</td>
        <td>CUB</td>
    </tr>
    <tr>
        <td>Meta-test</td>
        <td>miniImageNet</td>
        <td>tieredImageNet</td>
        <td>CUB</td>
        <td>Cars</td>
    </tr>
    <tr>
        <td>MAML(1)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?31.11 \pm 0.21" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?34.14 \pm 0.29" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?26.27 \pm 0.10" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?31.08 \pm 0.18" /></td>
    </tr>
    <tr>
        <td>BOIL(1)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?35.11 \pm 0.27" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?37.88 \pm 0.23" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?33.13 \pm 0.29" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?34.51 \pm 0.13" /></td>
    </tr>
    <tr>
        <td>MAML(5)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?38.74 \pm 0.17" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?42.11 \pm 0.23" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?30.50 \pm 0.21" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?39.74 \pm 0.19" /></td>
    </tr>
    <tr>
        <td>BOIL(5)</td>
        <td><img src="https://latex.codecogs.com/gif.latex?47.63 \pm 0.29" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?49.96 \pm 0.10" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?42.52 \pm 0.12" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?43.73 \pm 0.23" /></td>
    </tr>
</table>

### 3. 5-Way 5-shot test accuracy (%) of ResNet-12.

<table align="center" style="margin: 0px auto;">
    <tr>
        <td>Meta-train</td>
        <td colspan="3">miniImageNet</td>
        <td colspan="3">CUB</td>
    </tr>
    <tr>
        <td>Meta-test</td>
        <td>miniImageNet</td>
        <td>tieredImageNet</td>
        <td>CUB</td>
        <td>CUB</td>
        <td>miniImageNet</td>
        <td>Cars</td>
    </tr>
    <tr>
        <td>MAML w/ lsc</td>
        <td><img src="https://latex.codecogs.com/gif.latex?67.96 \pm 0.28" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?71.56 \pm 0.29" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?55.61 \pm 0.43" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?77.51 \pm 0.17" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?42.34 \pm 0.16" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?37.97 \pm 0.29" /></td>
    </tr>
    <tr>
        <td>MAML w/o lsc</td>
        <td><img src="https://latex.codecogs.com/gif.latex?66.03 \pm 0.18" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?69.43 \pm 0.22" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?52.10 \pm 0.21" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?70.90 \pm 0.31" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?37.32 \pm 0.25" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?33.94 \pm 0.31" /></td>
    </tr>
    <tr>
        <td>BOIL w/ lsc</td>
        <td><img src="https://latex.codecogs.com/gif.latex?69.68 \pm 0.25" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?71.43 \pm 0.38" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?61.00 \pm 0.36" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?81.54 \pm 0.14" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?44.54 \pm 0.20" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?40.05 \pm 0.39" /></td>
    </tr>
    <tr>
        <td>BOIL w/o lsc</td>
        <td><img src="https://latex.codecogs.com/gif.latex?70.90 \pm 0.20" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?74.29 \pm 0.31" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?61.83 \pm 0.49" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?83.23 \pm 0.14" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?44.62 \pm 0.10" /></td>
        <td><img src="https://latex.codecogs.com/gif.latex?40.86 \pm 0.35" /></td>
    </tr>
</table>