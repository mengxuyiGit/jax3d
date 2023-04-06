#!/usr/bin/env bash

EXP_SUFFIX=
python stage1.py --exp_suffix\
&& python stage2.py && python stage3.py