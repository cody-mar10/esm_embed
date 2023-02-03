#!/bin/bash

# simple executable for CHTC
CHECKPOINT=$1
OUTDIR=$2
FASTA=$3


set -e
ENVNAME="esm"
TARBALL="${ENVNAME}.tar.gz"
ENVDIR=$ENVNAME
STAGING="/staging/groups/anantharaman_group"
USER="ccmartin6"

cp $STAGING/$USER/$CHECKPOINT .
tar -xzf $CHECKPOINT
rm $CHECKPOINT

cp $STAGING/$USER/$FASTA .

##### CONDA
cp $STAGING/conda_envs/$TARBALL .
export PATH
mkdir $ENVDIR
tar -xzf $TARBALL -C $ENVDIR
. $ENVDIR/bin/activate
rm $TARBALL

python3 esm_embed.py -i $FASTA -th . -b 1024 -o $OUTDIR
rm $FASTA

# TODO: generalize this
tar -czf $OUTDIR.tar.gz $OUTDIR
mv $OUTDIR.tar.gz $STAGING/$USER

rm -rf $ENVDIR $OUTDIR checkpoints