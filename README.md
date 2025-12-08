# Conformal Optimistic Prediction

This repository contains the data and code required to reproduce the results in paper **"Distribution-informed Online Conformal Prediction"**.


## Environment
<p>
Please clone this repo and run following command locally for install the environment:
<pre>
conda create --name cop
pip install -r requirements.txt
</pre>
</p>



## Demo

Please run following command locally for local test:
<pre>
cd tests
python base_test.py configs/AMZN_test.yaml
python base_plots.py results/AMZN_test.pkl
</pre>

The plot results will be saved in <code>test/plots</code> folder. More commands can be seen in <code>tests/expbook.ipynb</code>. Users can modify the YAML files in the <code>configs</code> folder to configure the experimental settings.


## Implemented Methods

In addition to our proposed method, this repository provides implementations of several state-of-the-art online conformal prediction algorithms for benchmarking and reproducibility. The detailed information can be seen in our paper.

- ACI (Adaptive Conformal Inference)

- OGD (Online Gradient Descent)

- SF-OGD (Scale-Free OGD)

- Decay-OGD (OGD with decaying step sizes)

- Conformal PID (proportional-integral-derivative)

- ECI (Error-quantified Conformal Inference)

- LQT (Linear Quantile Tracking)



