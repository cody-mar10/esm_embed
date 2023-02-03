#!/bin/bash

# simple executable for CHTC

set -e
ENVNAME="esm"
TARBALL = "${ENVNAME}.tar.gz"
ENVDIR=$ENVNAME

cp $STAGING/$USER/checkpoints.tar.gz .
tar -xzf checkpoints.tar.gz

##### CONDA
cp $STAGING/conda_envs/$TARBALL .
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

python3 esm_embed.py -i checkpoints/test_1M.faa -th .

# TODO: generalize this
tar -czf out.tar.gz out
mv out.tar.gz $STAGING/$USER

rm -rf $ENVDIR $TARBALL checkpoints.tar.gz out