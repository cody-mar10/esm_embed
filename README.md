# esm_embed

Generating [ESM2](https://github.com/facebookresearch/esm) protein embeddings. The ESM team has provided a utility script to do this (`esm-extract`) that was not originally available when we made this repository.

**Note**: We have integrated this into the overall [Protein Set Transformer](https://github.com/AnantharamanLab/protein_set_transformer) workflow for an end-to-end pipeline to convert protein FASTA files to protein embeddings to genome embeddings. This repository is archived for reproducibility into the history of the PST paper.

## Installation

### Without GPUs

```bash
# in my experience, conda always handles pytorch installation better than pip
mamba create -n esm -c pytorch -c conda-forge 'pytorch>=2.0' 'python<3.12' cpuonly

mamba activate esm

pip install git+https://github.com/cody-mar10/esm_embed.git
```

### With GPUs

```bash
# in my experience, conda always handles pytorch installation better than pip
mamba create -n esm -c pytorch -c nvidia -c conda-forge 'pytorch>=2.0' 'python<3.12' pytorch-cuda=11.8

mamba activate esm

pip install git+https://github.com/cody-mar10/esm_embed.git
```

Installing this repository will create an executable called `esm-embed` to use for embedding protein FASTA files with ESM2 models.

## Usage

The bare minimum arguments to the `esm-embed` executable are:

```bash
esm-embed \
    --input FASTAFILE \
    --outdir OUTDIR \
    --esm ESM MODEL
```

To specify the locations of the input FASTA file, the output directory, and which ESM2 model you want. See the `-h` help page for allowed arguments. The corresponding information about each model can be found in the ESM [repository](https://github.com/facebookresearch/esm?tab=readme-ov-file#available).

Other arguments are for controlling the computational resources used:

- `--devices` number of GPUs or CPU threads
- `--accelerator` GPU or CPU, defaults to autodetecting
- `--precision` floating point precision for output embeddings. It is not recommended to use 64-bit due to the storage and memory required.

You can also specify where the ESM model will be downloaded (or where it was downloaded) using the `--torch-hub` argument.

### Output format

The output of `esm-embed` is a `.h5` with the field `data` that stores the embedding for each protein in the input FASTA file **IN THE SAME ORDER** as the FASTA file.

If you would like to install additional libraries to work with `.h5` files, you can also install this repository using:

```bash
pip install "esm_embed[h5] @ git+https://github.com/cody-mar10/esm_embed.git"
```

which will install the `pytables` package.

## Test run

We have provided a 10-sequence protein FASTA file for a test run:

```bash
esm-embed --input test/test.faa --outdir test/test_output --esm esm2_t6_8M --torch-hub test
```

The output embeddings have been provided for you to compare with.
