function movie = raw_frames_to_movie(frames)

for i = 1:length(frames) % iterate through all test examples 
    current_test = frames(i).bMsg;
    for j = 1:length(current_test) % iterate through every raw frame in current test example
        current_frame = current_test{j};
        movie(i,:,:,j) = flip(reshape(current_frame.Data,[current_frame.Width current_frame.Height])');
    end
end
end