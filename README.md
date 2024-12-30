## EFKAN: A KAN-Integrated Neural Operator For Efficient Magnetotelluric Forward Modeling

### Requirements
- PyTorch >=1.8.0
- torchinfo
- yaml
- numpy
- scipy
- ray

### files

- run: containing excutable python files
    - `efno_field_2d.py` is the EFNO for predicting electirc and magnetic field
    - `efno_2d.py` is the EFNO for predicting apparent resistivity and phase
    - `fourier_2d.py` is the Fourier Neural Operator([Li et al., 2021](https://arxiv.org/abs/2010.08895))
    - `cofigure.yml` is the configuration for `efno_2d.py`
    - `cofigure.yml` is the configuration for `efno__field2d.py`
- scripts: some auxiliary python fiels
- model: trained model file
- Log: log file
- temp: if stop early, you can file model file here.
