function [P,obj] = UEAF(X,W,Lv,Ind_ms,lambda1,lambda2,lambda3,max_iter,dim,r)
% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me: jiewen_pr@126.com 
% If you find the code is useful, please cite the following reference:
% Jie Wen , Zheng Zhang, Yong Xu, Bob Zhang, Lunke Fei, Hong Liu, 
% Unified Embedding Alignment with Missing Views Inferring for Incomplete Multi-View Clustering, 
% in Proc. of The Thirty-Third AAAI Conference on Artificial Intelligence, 2019.
% homepage: https://sites.google.com/view/jerry-wen-hit/publications
% ---------------- initialization --------------- %
alpha = ones(length(X),1)/length(X);
PPP = 0;
for iv = 1:length(X)
   rand('seed',iv*100);
   linshi_U = rand(size(X{iv},1),dim);
   if size(X{iv},1) > dim
        U{iv} = orth(linshi_U);
   else
       U{iv} = (orth(linshi_U'))';
   end
   rand('seed',iv*1000);
   E{iv} = rand(size(X{iv},1),size(W{iv},1));
   PPP = PPP+U{iv}'*(X{iv}+E{iv}*W{iv});
   inv_Lv{iv} = inv(lambda1*Lv{iv}+eye(size(Lv{iv},1)));
end
clear linshi_U
P = PPP/length(X);
alpha_r = alpha.^r;
% -----------  ³õÊ¼»¯  S F------------ %
rand('seed',666);
S = rand(size(X{1},2),size(X{1},2));
S = S - diag(diag(S));
S = (S'+S)/2;
Ls = diag(sum(S,2))-S;
[F, ~, ev] = eig1(Ls,dim, 0);
S2 = S.^2;
S2 = (S2+S2')/2;
Ls2 = diag(sum(S2,2))-S2;
for iter = 1:max_iter
    % ------------- P -------------- %
    Temp1 = 0;
    for iv = 1:length(X)
        Temp1 = Temp1+alpha_r(iv)*U{iv}'*(X{iv}+E{iv}*W{iv});
    end
    
    P = Temp1/(eye(size(Ls,1))+lambda2*Ls2)/sum(alpha_r);
    P(isnan(P)) = 0;
    P(isinf(P)) = 1e5;
    % ------------- S --------------- %
    sum_P = sum(P.^2, 1);
    Dp = bsxfun(@plus, sum_P, bsxfun(@plus, sum_P', -2 * (P' * P)));
    Dp = Dp - diag(diag(Dp));   
    
    sum_F = sum(F.^2, 2);
    Df = bsxfun(@plus, sum_F, bsxfun(@plus, sum_F', -2 * (F * F')));
    Df = Df - diag(diag(Df)); 
    S = -0.5*lambda3/lambda2*(Df./(max(Dp,1e-10)))/sum(alpha_r);
    S = S - diag(diag(S));
    for is = 1:size(S,1)
       ind = [1:size(S,1)];
       ind(is) = [];
       S(is,ind) = EProjSimplex_new(S(is,ind));
    end
    % --------------- F --------------- %
    LS = (S+S')/2;
    LS = diag(sum(LS)) - LS;
%     F_old = F;
    [F, ~, ev] = eig1(LS, dim, 0);
    % -------- U{v} E{v}--------------- %
    NormX = 0;
    S2 = S.^2;
    S2 = (S2+S2')/2;
    Ls2 = diag(sum(S2,2))-S2;
    for iv = 1:length(X)
       % -------- U{v} --------- %
       linshi = X{iv}+E{iv}*W{iv};
       temp = linshi*P';
       temp(isnan(temp)) = 0;
       temp(isinf(temp)) = 1e10;
       [Gs,~,Vs] = svd(temp,'econ');
       Gs(isnan(Gs)) = 0;
       Vs(isnan(Vs)) = 0;
       U{iv} = Gs*Vs'; 
       clear Gs Vs
       % ------- E{v} ------- %
       linshi  = U{iv}*P;
       linshi1 = linshi(:,Ind_ms{iv});
       E{iv} = inv_Lv{iv}*linshi1;
       
       % ------- obj reconstructed error ------------ %
       Rec_error(iv) = norm(X{iv}+E{iv}*W{iv}-U{iv}*P,'fro')^2+lambda1*trace(E{iv}'*Lv{iv}*E{iv}+0.5*lambda2*sum(sum(Dp.*S2)));
       NormX = NormX + norm(X{iv},'fro')^2;
    end
    % -------update alpha -------- %
    H = bsxfun(@power,Rec_error, 1/(1-r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,H,sum(H)); % alpha = H./sum(H);
    alpha_r = alpha.^r;
    % -------- obj ------------ %
    obj(iter) = (alpha_r*Rec_error'+lambda3*trace(F'*LS*F))/NormX;
    if iter > 2 && abs(obj(iter)-obj(iter-1))<1e-5
        iter
        break;
    end
end
end