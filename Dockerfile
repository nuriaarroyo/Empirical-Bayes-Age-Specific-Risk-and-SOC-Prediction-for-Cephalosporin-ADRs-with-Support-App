# Use a Miniconda base image
FROM continuumio/miniconda3:24.3.0-0

# Use bash as default shell for RUN
SHELL ["bash", "-lc"]

# Create working directory inside the container
WORKDIR /workspace

# Copy only the environment spec first (better build cache)
COPY environment.yml /tmp/environment.yml

# Create the conda environment inside the image
# Adjust the name "cephalo" if your environment.yml uses another name
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Make that environment the default Python
ENV CONDA_DEFAULT_ENV=cephalo
ENV PATH=/opt/conda/envs/cephalo/bin:$PATH

# Copy your whole project into the image (can be overridden by a bind mount)
COPY . /workspace

# Default command; docker-compose will override if needed
CMD ["bash"]
