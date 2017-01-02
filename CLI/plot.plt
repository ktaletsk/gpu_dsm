set term png size 1024,768
set output "Z_distr.png"
plot "distr_Z_arm_1.dat" with lines, "distr_Z_arm_2.dat" with lines, "distr_Z_arm_3.dat" with lines, "distr_Z_initial_arm_1.dat" with lines, "distr_Z_initial_arm_2.dat" with lines, "distr_Z_initial_arm_3.dat" with lines
set output "N_distr.png"
plot "distr_N_arm_1.dat" with lines, "distr_N_arm_2.dat" with lines, "distr_N_arm_3.dat" with lines, "distr_N_initial_arm_1.dat" with lines, "distr_N_initial_arm_2.dat" with lines, "distr_N_initial_arm_3.dat" with lines
set output "N_bp_distr.png"
plot "distr_N_arm_1_bp.dat" with lines, "distr_N_arm_2_bp.dat" with lines, "distr_N_arm_3_bp.dat" with lines, "distr_N_initial_arm_1_bp.dat" with lines, "distr_N_initial_arm_2_bp.dat" with lines, "distr_N_initial_arm_3_bp.dat" with lines
set output "Q_distr.png"
plot "distr_Q_arm_1.dat" with lines, "distr_Q_arm_2.dat" with lines, "distr_Q_arm_3.dat" with lines, "distr_Q_initial_arm_1.dat" with lines, "distr_Q_initial_arm_2.dat" with lines,    "distr_Q_initial_arm_3.dat" with lines
set output "Q_bp_distr.png"
plot "distr_Q_arm_1_bp.dat" with lines, "distr_Q_arm_2_bp.dat" with lines, "distr_Q_arm_3_bp.dat" with lines, "distr_Q_initial_arm_1_bp.dat" with lines, "distr_Q_initial_arm_2_bp.dat" with lines,    "distr_Q_initial_arm_3_bp.dat" with lines
