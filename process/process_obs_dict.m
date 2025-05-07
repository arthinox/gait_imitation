clear all;
clc;

import_fname = 'imitate_walk_train_1_obs.csv';
table = readtable(import_fname);

% com velocity in y direction
com_vel = table.com_vel_0_1;
t = table.t_0;

hold on
plot(t, com_vel)
plot(t, yline(1.25,'-r'))
legend('Actual COM Vel.','Target COM Vel.')
hold off