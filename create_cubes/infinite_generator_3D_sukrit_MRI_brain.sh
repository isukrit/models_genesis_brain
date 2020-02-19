#!/usr/bin/env python

for subset in `seq 1 9`
do
python -W ignore infinite_generator_3D_sukrit_MRI_brain.py \
--fold $subset \
--scale 32 \
--data ../ATLAS_data \
--save generated_cubes
done