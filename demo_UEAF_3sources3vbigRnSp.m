% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me: jiewen_pr@126.com 
% If you find the code is useful, please cite the following reference:
% Jie Wen , Zheng Zhang, Yong Xu, Bob Zhang, Lunke Fei, Hong Liu, 
% Unified Embedding Alignment with Missing Views Inferring for Incomplete Multi-View Clustering, 
% in Proc. of The Thirty-Third AAAI Conference on Artificial Intelligence, 2019.
% homepage: https://sites.google.com/view/jerry-wen-hit/publications

clear;
clc
Dataname = '3sources3vbigRnSp';

% % % % ---- parameters for percentDel = 0.1 --------- %
% % percentDel = 0.1;    
% % lambda1  = 1e-1;
% % lambda2  = 1e-2;
% % lambda3  = 1e-5;

% % % ---- parameters for percentDel = 0.3 --------- %
percentDel = 0.3;    
lambda1  = 1e-2;
lambda2  = 1e-1;
lambda3  = 1e-5;
% % % % ---- parameters for percentDel = 0.5 --------- %
% percentDel = 0.5;    
% lambda1  = 1e-5;
% lambda2  = 1e1;
% lambda3  = 1e-5;
Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Dataname);
load(Datafold);
    
num_view = length(X);
numClust = length(unique(truth));
numInst  = length(truth); 

neighbor = 7;
for f = 1:5     % you can choose it from 1~20  indicates randomly pre-formed incomoplete index
    ind_folds = folds{f};
    load(Dataname);
    truthF = truth;
    clear truth
    for iv = 1:length(X)
        X1 = X{iv}';
        X1 = NormalizeFea(X1,1);
        ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
        X1(ind_0,:) = 0;    
        Y{iv} = X1';
        W1 = eye(numInst);
        ind_1 = find(ind_folds(:,iv) == 1);
        W1(ind_1,:) = [];
        W{iv} = W1;                            
        Ind_ms{iv} = ind_0;
    end
    clear X X1 W1 ind_0
    X = Y;
    clear Y      

    % ---------- nearest neighbor graph of feature construction ------------ %
    for iv = 1:length(X)
        options = [];
        options.NeighborMode = 'KNN';
        options.k = neighbor;
        options.WeightMode = 'Binary';      % Binary  HeatKernel
        Z1 = full(constructW(X{iv},options));
        Z1 = (Z1+Z1')/2;
        Lv{iv} = diag(sum(Z1,2))-Z1;
        clear Z1;
    end

    max_iter = 50;
    dim = numClust;
    r = 3;
    [P,obj] = UEAF(X,W,Lv,Ind_ms,lambda1,lambda2,lambda3,max_iter,dim,r);            
    P(isnan(P)) = 0;
    P(isinf(P)) = 1e5;
    new_F = P';
    norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
    % avoid divide by zero
    for i = 1:size(norm_mat,1)
        if (norm_mat(i,1)==0)
            norm_mat(i,:) = 1;
        end
    end
    new_F = new_F./norm_mat; 
    rand('seed',230);
    pre_labels    = kmeans(real(new_F),numClust,'emptyaction','singleton','replicates',20,'display','off');
    result = ClusteringMeasure(truthF,pre_labels);
    ACC(f) = result(1)*100;
    NMI(f) = result(2)*100;
    Pur(f) = result(3)*100;
end
mean_acc = mean(ACC)
mean_nmi = mean(NMI)
mean_pur = mean(Pur)