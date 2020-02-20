#!/usr/bin/env python

for subset in `seq 6 9`
do
python -W ignore infinite_generator_3D_sukrit_MRI_brain_classification.py \
--fold $subset \
--scale 32 \
--data ../ATLAS_data \
--save generated_cubes
done