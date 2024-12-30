## EFKAN: A KAN-Integrated Neural Operator For Efficient Magnetotelluric Forward Modeling
![替代文本](./eval/figs/EFKAN.pdf "EFKAN")

### Requirements
- PyTorch >=1.8.0
- torchinfo
- yaml
- numpy
- scipy
- ray

### Python scrpits
- `data_gen/model_random.py`: generating the random smooth conductivity model
- `data_gen/model_block.py`: generating the random smooth conductivity model with block anomiles
- `run/efno_mlp.py` : the EFNO for predicting apparent resistivity and phase
- `run/efno_mlpkan.py` : the EFKAN for predicting apparent resistivity and phase
- `run/fourier_2d.py`: the Fourier Neural Operator([Li et al., 2021](https://arxiv.org/abs/2010.08895))
- `run/cofigure.yml`: configurations for EFNO and EFKAN training.
- `eval/plot_loss.ipynb`: ploting the training loss curves
