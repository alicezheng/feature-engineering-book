# Start from a core stack version
FROM jupyter/scipy-notebook:b78201340369

# Install Jupyter Notebook's extensions
# RUN conda install -c conda-forge jupyter_contrib_nbextensions && \
# 	  jupyter nbextensions_configurator enable --user
	
# Install Jupyter Lab's extensions
RUN jupyter labextension install \
  @jupyterlab/toc@3.0.0
RUN jupyter lab build

# Install additional packages
# RUN pip install -q --no-cache-dir nltk==3.4.5
