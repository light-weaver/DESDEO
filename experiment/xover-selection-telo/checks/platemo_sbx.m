% Writes OperatorGA's offspring for the fixed parent pairs of sbx_libraries.py, with mutation off.
% octave --no-gui --quiet --path <PlatEMO>/Algorithms/Utility\ functions platemo_sbx.m ...
args = argv();
out = args{end-2};
proC = str2double(args{end-1});
disC = str2double(args{end});
rand('twister', 1);

p1 = [0.2 0.8 0.01 0.97 0.0 0.4 0.5 0.3];
p2 = [0.8 0.2 0.03 0.995 0.5 0.4 1.0 0.31];
N = 200000;
D = numel(p1);
Problem = struct('encoding', ones(1, D), 'lower', zeros(1, D), 'upper', ones(1, D));

% OperatorGA splits the pool in two halves and stacks the first offspring above the second.
Offspring = OperatorGA(Problem, [repmat(p1, N, 1); repmat(p2, N, 1)], {proC, disC, 0, 20});
fid = fopen(out, 'wb');
fwrite(fid, Offspring', 'double', 0, 'ieee-le');
fclose(fid);
