# Exploration of Sales :chart_with_upwards_trend:
Personal project for applying DS skills in EDA and commercial strategies applied to SKU. 
# Folder structure :file_folder:

This is the folder structure of the project:

```bash
.
├── LICENSE
├── README.md
├── config.yaml
├── data
├── requirements.txt
└── src
    ├── eda.ipynb
    ├── models.ipynb
    └── utils
        ├── Descenso2Pasos.py
        ├── PuntosInteriores.py
```

In the `src` folder, there are two jupyter notebooks, one for the EDA and the other for the models. The `utils` folder contains the implementation of the two optimization algorithms used in the models notebook for a Recommender System. In the root folder, there are 2 html files with the results of the notebooks.

It is important to add the data paths at the `config.yaml` file.

# Requirements :clipboard:

User must have access to the two datasets provided in the data folder and should be listed in the `config.yaml` file as follows:

```yaml
main:
  ...
data:
    sales: name1
    sku: name2
    ...
```

Create a conda environment, avctivate it and then install the requiered packages with the following commands:

```bash
conda env create -n exploration_sales_entropia python=3.8
conda activate exploration_sales_entropia
pip install -r requirements.txt
```