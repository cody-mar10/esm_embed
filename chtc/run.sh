#!/bin/bash

# simple executable for CHTC
MODEL=$1
OUTDIR=$2
FASTA=$3

set -e
CHECKPOINT="${MODEL}.tar.gz"
ENVNAME="esm"
TARBALL="${ENVNAME}.tar.gz"
ENVDIR=$ENVNAME

##### move data
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

##### run tool
esm_embed -i $FASTA -th . -b 1024 -o $OUTDIR -m $MODEL -a gpu

##### cleanup
rm $FASTA
tar -czf $OUTDIR.tar.gz $OUTDIR
mv $OUTDIR.tar.gz $STAGING/$USER
rm -rf $ENVDIR $OUTDIR checkpoints
