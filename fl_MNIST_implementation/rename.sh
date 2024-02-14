#!/bin/bash

# Change to the directory containing files
cd /home/ob3942/repos/TrustShield/fl_CIFAR_implementation/histories/try
for file in *.npy; do
	# Get the current filename without the extension
	mv $file "${file:0:32}g${file:33:40}${file:58}"
done

		
		
