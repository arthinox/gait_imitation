l_ankle = readmatrix('l_ankle_angle.csv');
l_knee = readmatrix('l_knee_angle.csv');
l_hip = readmatrix('l_hip_angle.csv');

r_ankle = readmatrix('r_ankle_angle.csv');
r_knee = readmatrix('r_knee_angle.csv');
r_hip = readmatrix('r_hip_angle.csv');

mean_ankle = mean(cat(3, l_ankle, r_ankle),3);
mean_knee = mean(cat(3, l_knee, r_knee),3);
mean_hip = mean(cat(3, l_hip, r_hip),3);

mean_ankle = -0.0174533*mean_ankle(1:end-1);
mean_knee = -0.0174533*mean_knee(1:end-1);
mean_hip = -0.0174533*mean_hip(1:end-1);

out_mat = horzcat(mean_ankle,mean_knee,mean_hip);

T = array2table(out_mat);
T.Properties.VariableNames(1:3) = {'ankle_angle','knee_angle','hip_angle'};
writetable(T,'joint_angle_data.csv');