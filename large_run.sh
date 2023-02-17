#!/bin/bash

# simple executable for CHTC
MODEL=$1
OUTDIR=$2
FASTA=$3
CPU=$4

set -e
CHECKPOINT=${MODEL}.tar.gz
ENVNAME="esm"
TARBALL="${ENVNAME}.tar.gz"
ENVDIR=$ENVNAME

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

python3 esm_embed.py -i $FASTA -th . -b 1024 -o $OUTDIR -m $MODEL
rm $FASTA

# OVER HALF TIME IS SPENT GZIPPING SO USE PIGZ INSTEAD
tar -I "pigz -p ${CPU}" -cf $OUTDIR.tar.gz $OUTDIR
mv $OUTDIR.tar.gz $STAGING/$USER

rm -rf $ENVDIR $OUTDIR checkpoints
