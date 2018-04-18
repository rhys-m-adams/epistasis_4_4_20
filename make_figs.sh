###### Find the monotonic transformations that maximizes the PWM. Must be the first step if we don't use log KD for PWM
python figure_monotonic_transformation_fit.py
python figure_simulated_fit.py

#figure 1
python figure_contour_scales.py
python figure_1.py
python figure_contour_scales_no_bound.py

#figure 2
python figure_2.py
python figure_distance_v_epistasis.py
python figure_selected_sign.py
python figure_z_epistasis_pos.py
python figure_2_optimized.py
python figure_Z_w_bound.py
python figure_clone_Z.py
python figure_fraction_v_noise.py
python figure_replicate_epistasis.py
python figure_replicate_Z.py

#figure 3
# Must run
# ./submit_jobs.sh
# before make figure 3. This was done on a slurm cluster and takes around >10,000 hours to run, so it has been commented out

#figure 3
python figure_3.py
python figure_biochemical_fit_p_vals.py
python figure_3_optimal.py