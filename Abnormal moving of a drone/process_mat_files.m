function [frames,IMU_data,IMU_diff_pressure,IMU_mag,IMU_static_pressure,...
    IMU_temperature_baro,global_position_rel_alt,global_position_compass_hdg,...
    local_position_pose,local_position_odom] = process_mat_files(directory)
% LOADS DATA FROM MAT FILES
frames = [];
IMU_data=[];IMU_diff_pressure=[];IMU_mag=[];IMU_static_pressure=[];IMU_temperature_baro=[];
global_position_rel_alt=[];global_position_compass_hdg=[];local_position_pose=[];local_position_odom=[];
subdirs = dir(directory); % list of all subdirs in directory
for i = 3:length(subdirs) 
    subdir_full_path = fullfile(directory,subdirs(i).name);
    filenames = dir(fullfile(subdir_full_path , '*.mat')); % read all files with mat extesnsion
    total_mat_files = numel(filenames);   
    for j = 1:total_mat_files
        full_name = fullfile(subdir_full_path, filenames(j).name); % specify image name with full path and extension      
        current_sensor_data = load(full_name);
        % extracting data from camera
        if (filenames(j).name == "_pylon_camera_node_image_raw.mat")
            frames = [frames, current_sensor_data];
        end
        % ectracting data from IMU 
        if (filenames(j).name == "_mavros_imu_data.mat")
            IMU_data = [IMU_data, current_sensor_data];
        end
        if (filenames(j).name == "_mavros_imu_diff_pressure.mat")
            IMU_diff_pressure = [IMU_diff_pressure, current_sensor_data];
        end
        if (filenames(j).name == "_mavros_imu_mag.mat")
            IMU_mag = [IMU_mag, current_sensor_data];
        end
        if (filenames(j).name == "_mavros_imu_static_pressure.mat")
            IMU_static_pressure = [IMU_static_pressure, current_sensor_data];
        end
        if (filenames(j).name == "_mavros_imu_static_pressure.mat")
            IMU_temperature_baro = [IMU_temperature_baro, current_sensor_data];
        end
        % extracting data from global_position
        if (filenames(j).name == "_mavros_global_position_rel_alt.mat")
            global_position_rel_alt = [global_position_rel_alt, current_sensor_data];
        end
        if (filenames(j).name == "_mavros_global_position_compass_hdg.mat")
            global_position_compass_hdg = [global_position_compass_hdg, current_sensor_data];
        end
        % ectracting data from local_position
        if (filenames(j).name == "_mavros_local_position_pose.mat")
            local_position_pose = [local_position_pose, current_sensor_data];
        end
        if (filenames(j).name == "_mavros_local_position_odom.mat")
            local_position_odom = [local_position_odom, current_sensor_data];
        end
    end
end
end