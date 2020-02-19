#!/usr/bin/env python
for subset in `seq 1 10`
do
python -W ignore infinite_generator_3D_sukrit_CT_brain.py \
--fold $subset \
--scale 32 \
--data ../CQ500_data \
--save generated_cubes_CT
done