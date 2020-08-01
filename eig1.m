function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
% isMax=1����ѡȡA���������ֵ��Ӧ����������
% isMax=0����ѡȡA����С����ֵ��Ӧ����������

if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
end;
% [v d] = eig(A);

% -------------------- ����1 -------------------------- %
% % �����޸İ취���� http://ask.cvxr.com/t/eig-did-not-converge-in-prox-trace/996/16
A(isnan(A)) = 0;
A(isinf(A)) = 1e5;
try
    [v,d] = eig(A);
catch ME
    if (strcmpi(ME.identifier,'MATLAB:eig:NoConvergence'))
        [v,d] = eig(A, eye(size(A)));
    else
        rethrow(ME);
    end
end

% % % --------------------- ����2 ��SVD�滻eig----------------------- %
% % [v,d,w] = svd(A);
% % d = d.*sign(diag(real(dot(v,w,1))));

d = diag(d);
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d);                % ����
else
    [d1, idx] = sort(d,'descend');      % ����
end;    

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);