clc;clear;close all;
dir_in='F:/Prenatal_Opioid_exposure/TimeCourse/Harvard';
result_dir='./Classify_results/eFC';
mkdir (result_dir)
load 'F:/Prenatal_Opioid_exposure/FD_left.mat'

ss=15; sc=26;
sz = [41 3];
varTypes = {'double','string','double'};
varNames = {'label','name','FD'};
sub = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
nROI=112;
% %abstract the appropriate subjects 
count=0;
for i=1:ss
    file = sprintf('%s/%s_%s.txt',dir_in,FD_left{i,1}, FD_left{i,2}(1:end-7)); 
    TC=load (file); 
    if size(TC,1)==500 && size(TC,2)==112
        count=count+1;
        BOLD{count,1}=TC(11:end,:);
        sub.label(count,1)=-1;
        sub.name{count,1}=char(FD_left{i,1});
        sub.FD(count,1)=FD_left{i,3};
    end
end
num_con=count;

for i=1:sc
    file = sprintf('%s/%s_%s.txt',dir_in,FD_left{num_con+i,1}, FD_left{num_con+i,2}(1:end-7)); 
    TC=load (file); 
    if size(TC,1)==500 && size(TC,2)==112
        count=count+1;
        BOLD{count,1}=TC(11:end,:);
        sub.label(count,1)=1;
        sub.name{count,1}=char(FD_left{num_con+i,1});
        sub.FD(count,1)=FD_left{num_con+i,3};
    end
end
num_opoid=count-num_con;
save (char(strcat(result_dir,'/classify_data.mat')), 'BOLD', 'sub'); 

%==============================================================================================
%construct edge_centric networks
s=1;  % step size for dynamic FC calculation
W=50:10:120;  % window length
C=100:100:800;  % number of clusters
lambda_lasso=0.05;  % hyper-parameter in lasso, controls feature selection
nSubj=length(BOLD);
num_C=length(C);
num_W=length(W);

for i=1:num_W % number of clusters   parfor
    for j=1:num_C
        [BrainNet{i,j},IDX{i,j}]=dHOFC(BOLD,W(i),s,C(j));
    end
end
BrainNet=reshape(BrainNet,1,num_W*num_C);
save (char(strcat(result_dir,'/eFC_net.mat')),'BrainNet','IDX','-v7.3');

%==============================================================================================
% Extract features from FC networks under different parameters
for i=1:length(BrainNet)
    temp=ceil(i/length(W));
    Feat=zeros(nSubj,C(temp)); 
    flag=2;
    for j=1:nSubj
        Feat(j,:)=wlcc(BrainNet{i}(:,:,j),flag);
    end
    All_Feat{i} = Feat;
end
save (char(strcat(result_dir,'/All_Feat.mat')),'All_Feat','-v7.3'); 
%==============================================================================================
%classification: 10 times 10-fold cross-validation 
label=sub.label;
fold_times=10;
kfoldout=10;
para_test_flag=1;
rng('shuffle');
for i=1:fold_times
    fprintf(1,'Begin 10-fold cross-validation time %d...\n',i);
    cpred = zeros(nSubj,1);
    acc = zeros(nSubj,1);
    score = zeros(nSubj,1);
    Test_res=zeros(1,kfoldout);
    c_out = cvpartition(nSubj,'k', kfoldout);
    kfoldin=kfoldout-1;
    % 10-fold cross-validation for testing
    for fdout=1:c_out.NumTestSets
        fprintf(1,'Begin process %d%%...\n',fdout*10);
        % Nested CV on Train data for Model selection
        max_acc = 0;
        for t = 1:length(All_Feat)
            % Feature generation
            Feat = All_Feat{t};
            Train_data=Feat(training(c_out,fdout),:);
            Train_lab=label(training(c_out,fdout));
            %         Test_data=Feat(test(c_out,fdout),:);
            %         Test_lab=label(test(c_out,fdout));
            tmpTestCorr = zeros(length(Train_lab),1);
            c_in=cvpartition(length(Train_lab),'k',kfoldin);
            for fdin=1:c_in.NumTestSets
                InTrain_data=Train_data(training(c_in,fdin),:);
                InTrain_lab=Train_lab(training(c_in,fdin));
                Vali_data=Train_data(test(c_in,fdin),:);
                Vali_lab=Train_lab(test(c_in,fdin));

                % Feature selection using lasso
                midw=lasso(InTrain_data,InTrain_lab,'Lambda',lambda_lasso);  % parameter lambda for sparsity
                InTrain_data=InTrain_data(:,midw~=0);
                Vali_data=Vali_data(:,midw~=0);

                % Feature normalization
                Mtr=mean(InTrain_data);
                Str=std(InTrain_data);
                InTrain_data=InTrain_data-repmat(Mtr,size(InTrain_data,1),1);
                InTrain_data=InTrain_data./repmat(Str,size(InTrain_data,1),1);
                Vali_data=Vali_data-repmat(Mtr,size(Vali_data,1),1);
                Vali_data=Vali_data./repmat(Str,size(Vali_data,1),1);

                % train SVM model
                classmodel=svmtrain(InTrain_lab,InTrain_data,'-t 0 -c 1 -q'); % linear SVM (require LIBSVM toolbox)
                % classify
                [~,acc,~]=svmpredict(Vali_lab,Vali_data,classmodel,'-q');
                tmpTestCorr(fdin,1) = acc(1);
            end
            mTestCorr = sum(tmpTestCorr)/length(tmpTestCorr);
            if mTestCorr>max_acc
                max_acc = mTestCorr;
                opt_t(i,fdout) = t;
            end
        end
        
        % Feature generation
        Feat = All_Feat{opt_t(i,fdout)};
        Train_data = Feat(training(c_out,fdout),:);
        Train_lab = label(training(c_out,fdout));
        Test_data=Feat(test(c_out,fdout),:);
        Test_lab=label(test(c_out,fdout));

        % Feature selection 
        midw=lasso(Train_data,Train_lab,'Lambda',lambda_lasso);  % parameter lambda for sparsity
        Train_data=Train_data(:,midw~=0);
        Test_data=Test_data(:,midw~=0);
        feature_index_lasso{i,fdout}=midw;
      
        % Feature normalization 
        Mtr=mean(Train_data);
        Str=std(Train_data);
        Train_data=Train_data-repmat(Mtr,size(Train_data,1),1);
        Train_data=Train_data./repmat(Str,size(Train_data,1),1);
        Test_data=Test_data-repmat(Mtr,size(Test_data,1),1);
        Test_data=Test_data./repmat(Str,size(Test_data,1),1);

        % train SVM model 
        classmodel=svmtrain(Train_lab,Train_data,'-t 0 -c 1 -q'); % linear SVM (require LIBSVM toolbox)
        % classify
        w{fdout}=classmodel.SVs'*classmodel.sv_coef;
        [cpred(test(c_out,fdout)),~,score(test(c_out,fdout))]=svmpredict(Test_lab,Test_data,classmodel,'-q');
    end

    Acc=100*sum(cpred==label)/nSubj;
    [AUC(i),SEN(i),SPE(i),F1(i),Youden(i),BalanceAccuracy(i),plot_ROC{i}]=perfeval_kfold(label,cpred,score);
end

rng('default');
AUC=mean(AUC);
SEN=mean(SEN);
SPE=mean(SPE);
F1=mean(F1);
Acc=mean(Acc);
Youden=mean(Youden);
BalanceAccuracy=mean(BalanceAccuracy);

fprintf(1,'Testing result AUC: %g\n',AUC);
fprintf(1,'Testing result Sens: %3.2f%%\n',SEN);
fprintf(1,'Testing result Spec: %3.2f%%\n',SPE);
fprintf(1,'Testing result Youden: %3.2f%%\n',Youden);
fprintf(1,'Testing result F-score: %3.2f%%\n',F1);
fprintf(1,'Testing result BAC: %3.2f%%\n',BalanceAccuracy);
ROC_kfold(plot_ROC,result_dir,fold_times);

% identify classifying features
midw_lasso=feature_index_lasso;
[result_features]=back_find_high_node(W,C,nROI,w,midw_lasso,IDX,opt_t);

%==============================================================================================
%discriminate FC
N=112;
FC=[];
wei=[];
for i=1:length(result_features)
    FC=[FC; result_features{i,2}];
    wei=[wei;ones(length(result_features{i,2}),1)*result_features{i,4}*result_features{i,1}/100];
end
    
FC_inf=[FC wei];
FC_inf=sortrows(FC_inf,[1 2]);

%combine volume 1 and 2 as volume 4 to indicate the FC
for i=1:length(FC_inf)-1
    FC_inf(i,4)=str2num([num2str(FC_inf(i,1)) num2str(FC_inf(i,2))]);
end
FC_uni=unique(FC_inf(:,4));
count=1;
FC_inf_uni=[];
for i=1:length(FC_uni)
    ind=[];
    ind=find(FC_inf(:,4)==FC_uni(i));
    FC_inf_uni(count,[1:2, 4])=FC_inf(ind(1),[1:2,4]);
    FC_inf_uni(count,3)=mean(abs(FC_inf(ind,3))); %the results of abs equal no abs
    count=count+1;
end
FC_inf_uni=sortrows(FC_inf_uni,[1 2]);
save (char(strcat(result_dir,'/FC_inf.mat')),'FC_inf', 'FC_inf_uni'); 

%==============================================================================================
%plot the FC mat
mat=zeros(112,112);
for i=1:length(FC_inf_uni)
    mat(FC_inf_uni(i,1), FC_inf_uni(i,2))=FC_inf_uni(i,3);
    mat(FC_inf_uni(i,2), FC_inf_uni(i,1))=FC_inf_uni(i,3);
end
mat=abs(mat);
%mat(find(mat<0.1))=0;
imagesc(mat)
set(gca,'Fontname', 'Arial','fontsize',15);

%==============================================================================================
% discriminate ROI
for i=1:112
    ind=[];
    ind=find(FC_inf(:,1)==i);
    ind=[ind; find(FC_inf(:,2)==i)];
    wei_node(i,1)=sum(abs(FC_inf(ind,3)));
end

load F:\Prenatal_Opioid_exposure\process\HarvardOxford_inf.mat
percent=0.25;
BB = sort(wei_node,'descend');
thr = BB(floor(N*percent));
CC = wei_node;
CC(abs(CC)<thr) = 0;
index=find(CC>0)';
lab_N=HarvardOxford_inf(index, 4);
lab=HarvardOxford_inf(index, 1);
lab_abr=HarvardOxford_inf(index, 5);
wei=wei_node(index);

sig_ROI=table(lab_N,lab,lab_abr, wei);
save(char(strcat(result_dir,'/sig_ROI.mat')), 'sig_ROI');

